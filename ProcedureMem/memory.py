import json
import os
import random
import time
import numpy as np
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

from ProcedureMem.llm_api import (
    get_llm_response,
    get_embedding_model,
    resolve_memory_build_seed,
    resolve_memory_build_temperature,
    resolve_memory_build_top_k,
)
from ProcedureMem.Alfworld.memory_prompts import (
    generate_workflow_from_trajectory_prompt,
    generate_events_from_trajectory_prompt,
    generate_workflow_from_events_prompt,
    build_prompt_manifest,
    get_prompt_spec,
    prompt_manifest_mismatches,
)
from ProcedureMem.memory_utils import (
    compute_facts_embeddings,
    save_facts_embedding_cache,
    load_facts_embedding_cache,
    cosine_similarity
)
from ProcedureMem.memory_adjust import adjust_memory
from ProcedureMem.reranker import OpenMemReranker, format_workflow_candidate

class Memory:
    def __init__(self, **kwargs):
        self.is_cold_start = kwargs.get("is_cold_start", False)
        self.policy = kwargs.get("policy", {})
        self.traj_file_path = kwargs.get("traj_file_path", None)
        self.retrieve_num = kwargs.get("retrieve_num", 10)
        self.memory_dir = kwargs.get("memory_dir", "memory")
        self.prompt_domain = kwargs.get("prompt_domain", "alfworld")
        self.requested_build_model = (
            kwargs.get("build_model") or os.getenv("MEMORY_BUILD_MODEL_NAME")
        )
        self.build_model = self.requested_build_model or os.getenv("MODEL_NAME")
        if not self.build_model:
            raise RuntimeError(
                "Missing memory build model. Set MEMORY_BUILD_MODEL_NAME, pass "
                "build_model, or configure MODEL_NAME as a fallback."
            )
        self.build_temperature = resolve_memory_build_temperature(
            kwargs.get("build_temperature")
        )
        self.build_seed = resolve_memory_build_seed(kwargs.get("build_seed"))
        self.build_top_k = resolve_memory_build_top_k(kwargs.get("build_top_k"))
        self.build_api_key = (
            kwargs.get("build_api_key")
            or os.getenv("MEMORY_BUILD_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if "build_api_base_url" in kwargs:
            self.build_api_base_url = kwargs["build_api_base_url"] or None
        elif "MEMORY_BUILD_API_BASE_URL" in os.environ:
            self.build_api_base_url = (
                os.getenv("MEMORY_BUILD_API_BASE_URL") or None
            )
        else:
            self.build_api_base_url = (
                os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
            )
        self.trajectory_file = (
            os.path.abspath(self.traj_file_path) if self.traj_file_path else None
        )
        self.trajectory_count = None

        if self.prompt_domain != "alfworld":
            raise ValueError(
                f"Unsupported prompt domain: {self.prompt_domain}. "
                "This ALFWorld entry point requires prompt_domain='alfworld'."
            )

        self.memory_size = kwargs.get("memory_size", 1000)
        self.build_policy = self.policy.get("build")
        
        if self.build_policy not in ["round", "direct"]:
            raise ValueError(f"Invalid build policy: {self.build_policy}. Must be 'round' or 'direct'.")
        self.retrieve_policy = self.policy.get("retrieve")
        if self.retrieve_policy not in ["query", "facts", "random", "ave_fact"]:
            raise ValueError(f"Invalid retrieve policy: {self.retrieve_policy}. Must be 'query', 'facts', 'random', or 'ave_fact'.")
        self.update_policy = self.policy.get("update")
        self.cache_dir = self.memory_dir + "/vector_cache"
        self.facts_cache_path =  self.cache_dir + "/facts_embedding_cache.pkl"
        self.documents_path = self.memory_dir + "/" + self.build_policy + "/documents.json"
        self.manifest_path = self.memory_dir + "/" + self.build_policy + "/manifest.json"
        self.prompt_spec = get_prompt_spec(self.build_policy)
        self.documents = []
        self.vector_store = None
        self.doc_facts_embeddings = None


        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        os.makedirs(os.path.dirname(self.documents_path), exist_ok=True)

        # Initialize embedding model
        self.embedding = get_embedding_model()
        self._initialize_cached_embedder()

        # Load document metadata
        

        if self.is_cold_start:
            self._cold_start()
        



    def _initialize_cached_embedder(self):
        self.store = LocalFileStore(self.cache_dir)
        namespace = getattr(self.embedding, "model", None) or self.embedding.__class__.__name__
        self.cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.embedding, self.store, namespace=str(namespace)
        )

    def save_documents(self):
        """Persist workflow documents and their construction manifest."""
        with open(self.documents_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in self.documents],
                f, indent=2, ensure_ascii=False
            )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                build_prompt_manifest(
                    self.build_policy,
                    build_model=self.build_model,
                    build_temperature=self.build_temperature,
                    build_seed=self.build_seed,
                    build_top_k=self.build_top_k,
                    trajectory_file=self.trajectory_file,
                    trajectory_count=self.trajectory_count,
                ),
                f,
                indent=2,
                ensure_ascii=False,
            )

    def rebuild_index(self):
        """Rebuild retrieval state; an empty online memory is valid."""
        if not self.documents:
            self.vector_store = None
            self.doc_facts_embeddings = None
            return
        self.vector_store = FAISS.from_documents(self.documents, self.cached_embedder)
        self.doc_facts_embeddings = None
        if self.policy.get("retrieve") == "ave_fact":
            self.doc_facts_embeddings = load_facts_embedding_cache(self.facts_cache_path)
            if self.doc_facts_embeddings is None:
                print("[INFO] Computing facts embeddings for documents...")
                # Compute facts embeddings for all documents
                self.doc_facts_embeddings = compute_facts_embeddings(self.documents, self.embedding)
                save_facts_embedding_cache(self.facts_cache_path, self.doc_facts_embeddings)
            else:
                print(f"[INFO] Loaded facts embeddings from {self.facts_cache_path}")

    def append_documents(self, documents):
        """Append built documents without applying statistics, reflection, or eviction."""
        new_documents = list(documents)
        existing_ids = {
            doc.metadata.get("memory_id")
            for doc in self.documents
            if doc.metadata.get("memory_id") is not None
        }
        incoming_ids = [
            doc.metadata.get("memory_id")
            for doc in new_documents
            if doc.metadata.get("memory_id") is not None
        ]
        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("Cannot append duplicate memory IDs")
        repeated = sorted(set(incoming_ids) & existing_ids)
        if repeated:
            raise ValueError("Memory IDs already exist: " + ", ".join(repeated[:5]))
        self.documents.extend(new_documents)

    def _save_documents(self):
        """Backward-compatible save-and-rebuild operation."""
        self.save_documents()
        self.rebuild_index()

    def process_trajectory_item(self, d):
        """
        Process a single trajectory item.
        This function includes logic for checking existence, building workflow, and appending new documents.
        """
        source = d.get("source")
        query = d.get("query").split("\n\n")[0]
        trajectory = d.get("trajectory")
        facts = d.get("facts", {})

        memory_id = d.get("memory_id")
        if memory_id is not None:
            if any(doc.metadata.get("memory_id") == memory_id for doc in self.documents):
                print(f"[INFO] Memory ID '{memory_id}' already exists. Skipping...")
                return None
        elif any(
            doc.metadata.get("query") == query
            and doc.metadata.get("build_policy") == self.build_policy
            and doc.metadata.get("prompt_sha256") == self.prompt_spec.sha256
            for doc in self.documents
        ):
            print(f"[INFO] Query '{query}' with build policy '{self.build_policy}' already exists. Skipping...")
            return None

        # Build workflow
        workflow = self.build(query, trajectory)

        # Create Document
        extra_metadata = dict(d.get("metadata") or {})
        doc = Document(
            page_content=query,
            metadata={
                "source": source,
                "query": query,
                "workflow": workflow,
                "facts": facts,
                "build_policy": self.build_policy,
                "prompt_domain": self.prompt_spec.domain,
                "prompt_version": self.prompt_spec.version,
                "prompt_sha256": self.prompt_spec.sha256,
                "hit": 0,
                "success": 0,
                **extra_metadata,
            }
        )
        if memory_id is not None:
            doc.metadata["memory_id"] = str(memory_id)

        return doc

    def build_document(self, item):
        """Build one workflow Document without mutating or rebuilding memory."""
        return self.process_trajectory_item(item)

    def process_trajectory_item_reflect(self, trajectory, reward, workflow):
        if not reward and workflow != "":
            new_workflow = adjust_memory(worfklow=workflow, reward=reward, trajectory=trajectory)
            print(f"Original workflow: {workflow}")
            print(f"Adjusted workflow: {new_workflow}")
            for doc in self.documents:
                if doc.metadata.get("workflow") == workflow:
                    doc.metadata["workflow"] = new_workflow
                    break


    def _cold_start(self):
        """
        Cold start the memory by building it from the trajectory file.
        """

        if os.path.exists(self.documents_path):
            self._validate_prompt_manifest()
            with open(self.documents_path, "r", encoding="utf-8") as f:
                docs_data = json.load(f)
                self.documents = [Document(**d) for d in docs_data]
            self._validate_document_prompt_metadata()
            print(f"[INFO] Loaded {len(self.documents)} documents from {self.documents_path}")



        with open(self.traj_file_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
        if len(traj_data) > self.memory_size:
            traj_data = traj_data[:self.memory_size]
        current_trajectory_count = len(traj_data)

        new_documents = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_data = {executor.submit(self.process_trajectory_item, d): d for d in traj_data}
            for future in tqdm(as_completed(future_to_data), desc="Building memory from trajectory", total=len(traj_data)):
                try:
                    doc = future.result()
                    if doc:  # Only add non-None results
                        new_documents.append(doc)
                except Exception as e:
                    print(f"[ERROR] An error occurred while processing trajectory item: {e}")
        
        # Update documents list and save to disk
        self.documents.extend(new_documents)
        if self.trajectory_count is None or new_documents:
            self.trajectory_file = os.path.abspath(self.traj_file_path)
            self.trajectory_count = current_trajectory_count
        self._save_documents()

        print(f"[INFO] {len(new_documents)} new documents added.")

    def _validate_prompt_manifest(self):
        """Refuse to mix documents produced by a different or unknown prompt."""
        if not os.path.exists(self.manifest_path):
            raise RuntimeError(
                f"Memory cache {self.documents_path} has no prompt manifest. It may "
                "have been built with the legacy TravelPlanner prompt. Archive it or "
                "choose a new memory_dir before building ALFWorld memory."
            )

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("schema_version") != 2:
            raise RuntimeError(
                f"Memory manifest schema mismatch in {self.manifest_path}: "
                f"found {manifest.get('schema_version')!r}, expected 2. Rebuild the "
                "memory so its build model and trajectory source are recorded."
            )
        cached_prompt = manifest.get("prompt", {})
        expected_prompt = self.prompt_spec.as_dict()
        mismatches = prompt_manifest_mismatches(manifest, self.prompt_spec)
        if mismatches:
            details = ", ".join(
                f"{key}={cached_prompt.get(key)!r} (expected {expected_prompt.get(key)!r})"
                for key in mismatches
            )
            raise RuntimeError(
                f"Memory cache prompt mismatch in {self.manifest_path}: {details}. "
                "Archive the old cache or choose a new memory_dir; do not mix memories "
                "built from different prompts."
            )

        cached_build_model = manifest.get("build_model")
        if not cached_build_model:
            raise RuntimeError(
                f"Memory manifest {self.manifest_path} does not record build_model. "
                "Rebuild the memory in a new memory_dir."
            )
        if (
            self.requested_build_model
            and self.requested_build_model != cached_build_model
        ):
            raise RuntimeError(
                f"Memory cache was built by {cached_build_model!r}, but "
                f"{self.requested_build_model!r} was requested. Use the existing "
                "artifact without --memory-build-model, or choose a new memory_dir."
            )
        self.build_model = cached_build_model
        cached_build_temperature = manifest.get("build_temperature")
        if (
            cached_build_temperature is not None
            and float(cached_build_temperature) != self.build_temperature
        ):
            raise RuntimeError(
                f"Memory cache was built with temperature "
                f"{cached_build_temperature!r}, but {self.build_temperature!r} "
                "was requested. Use matching MEMORY_BUILD_TEMPERATURE or choose "
                "a new memory_dir."
            )
        cached_build_seed = manifest.get("build_seed")
        if cached_build_seed is not None and int(cached_build_seed) != self.build_seed:
            raise RuntimeError(
                f"Memory cache was built with seed {cached_build_seed!r}, but "
                f"{self.build_seed!r} was requested. Use matching "
                "MEMORY_BUILD_SEED or choose a new memory_dir."
            )
        cached_build_top_k = manifest.get("build_top_k")
        if (
            cached_build_top_k is not None
            and int(cached_build_top_k) != self.build_top_k
        ):
            raise RuntimeError(
                f"Memory cache was built with top_k {cached_build_top_k!r}, but "
                f"{self.build_top_k!r} was requested. Use matching "
                "MEMORY_BUILD_TOP_K or choose a new memory_dir."
            )
        self.trajectory_file = manifest.get("trajectory_file")
        self.trajectory_count = manifest.get("trajectory_count")

    def _validate_document_prompt_metadata(self):
        incompatible = [
            index
            for index, doc in enumerate(self.documents)
            if doc.metadata.get("prompt_domain") != self.prompt_spec.domain
            or doc.metadata.get("prompt_version") != self.prompt_spec.version
            or doc.metadata.get("prompt_sha256") != self.prompt_spec.sha256
        ]
        if incompatible:
            preview = ", ".join(str(index) for index in incompatible[:10])
            raise RuntimeError(
                f"Memory documents contain incompatible prompt metadata at indexes "
                f"{preview}. Archive {self.documents_path} or choose a new memory_dir."
            )

    def build(self, query, trajectory):
        """
        Build a workflow from the given query and trajectory based on the specified build policy.
        """
        # Generate workflow
        if self.build_policy == "round":
            events = get_llm_response(
                generate_events_from_trajectory_prompt(query, trajectory),
                is_string=False,
                model=self.build_model,
                api_key=self.build_api_key,
                api_base_url=self.build_api_base_url,
                temperature=self.build_temperature,
                seed=self.build_seed,
                top_k=self.build_top_k,
            )
            workflow_ids = get_llm_response(
                generate_workflow_from_events_prompt(query, events),
                is_string=False,
                model=self.build_model,
                api_key=self.build_api_key,
                api_base_url=self.build_api_base_url,
                temperature=self.build_temperature,
                seed=self.build_seed,
                top_k=self.build_top_k,
            )
            workflow = [events[wid - 1]['action'] for wid in workflow_ids]
        elif self.build_policy == "direct":
            workflow = get_llm_response(
                generate_workflow_from_trajectory_prompt(query, trajectory),
                is_string=True,
                model=self.build_model,
                api_key=self.build_api_key,
                api_base_url=self.build_api_base_url,
                temperature=self.build_temperature,
                seed=self.build_seed,
                top_k=self.build_top_k,
            )
        
        return workflow

    def retrieve(self, key):
        """
        Retrieve from memory according to the specified policy.
        """
        retrieve_num = min(self.retrieve_num, len(self.documents))
        if retrieve_num == 0 or self.vector_store is None:
            return []
        if self.retrieve_policy == "query":
            return self.vector_store.similarity_search_with_score(key, k=retrieve_num, score_threshold=0.5)

        elif self.retrieve_policy == "facts":
            key = str(key)
            return self.vector_store.similarity_search_with_score(key, k=retrieve_num, score_threshold=0.4)

        elif self.retrieve_policy == "random":
            return random.sample(self.documents, min(self.retrieve_num, len(self.documents)))

        elif self.retrieve_policy == "ave_fact":
            if not isinstance(key, dict):
                raise ValueError("For 'ave_fact' policy, key must be a dict of query facts.")

            query_facts_embeddings = {
                k: self.embedding.embed_query(str(v))
                for k, v in key.items()
            }

            scored_docs = []
            for doc in self.documents:
                doc_facts_embeddings = self.doc_facts_embeddings.get(doc.metadata["source"], {})
                common_keys = set(query_facts_embeddings) & set(doc_facts_embeddings)
                if not common_keys:
                    continue

                similarities = [
                    cosine_similarity(
                        np.array(query_facts_embeddings[ck]),
                        np.array(doc_facts_embeddings[ck])
                    )
                    for ck in common_keys
                ]

                avg_sim = float(np.mean(similarities))
                scored_docs.append((avg_sim, doc))

            scored_docs.sort(key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored_docs[:self.retrieve_num]]

        else:
            raise ValueError(f"Unknown retrieve policy: {self.retrieve_policy}")

    def retrieve_with_rerank(
        self,
        key: str,
        *,
        reranker: OpenMemReranker,
        candidate_k: int = 20,
        top_n: int = 10,
        score_threshold: float | None = None,
    ) -> dict[str, object]:
        """Retrieve workflow candidates with FAISS, then rerank them with OpenMem."""
        if self.retrieve_policy != "query":
            raise ValueError("Reranking currently requires retrieve policy 'query'")
        if candidate_k < 1 or top_n < 1:
            raise ValueError("candidate_k and top_n must be at least 1")
        if top_n > candidate_k:
            raise ValueError("top_n cannot exceed candidate_k")
        if score_threshold is not None and score_threshold < 0:
            raise ValueError("score_threshold must be non-negative")

        pipeline_started = time.perf_counter()
        search_started = time.perf_counter()
        search_kwargs = {"k": min(candidate_k, len(self.documents))}
        if score_threshold is not None:
            search_kwargs["score_threshold"] = score_threshold
        candidates = self.vector_store.similarity_search_with_score(
            key, **search_kwargs
        )
        candidate_search_latency_ms = (time.perf_counter() - search_started) * 1000.0
        if not candidates:
            raise RuntimeError("FAISS returned no workflow candidates for reranking")

        response = reranker.rerank(
            query=key,
            documents=[format_workflow_candidate(doc) for doc, _ in candidates],
            top_n=min(top_n, len(candidates)),
        )
        if not response.results:
            raise RuntimeError("OpenMem returned no reranked workflow items")
        reranked = []
        for rerank_rank, result in enumerate(response.results, start=1):
            document, vector_score = candidates[result.index]
            reranked.append(
                {
                    "document": document,
                    "vector_rank": result.index + 1,
                    "vector_score": float(vector_score),
                    "rerank_rank": rerank_rank,
                    "rerank_score": result.relevance_score,
                }
            )
        pipeline_latency_ms = (time.perf_counter() - pipeline_started) * 1000.0
        return {
            "candidates": candidates,
            "items": reranked,
            "candidate_search_latency_ms": candidate_search_latency_ms,
            "rerank_api_latency_ms": response.latency_ms,
            "rerank_pipeline_latency_ms": pipeline_latency_ms,
            "request_id": response.request_id,
            "prompt_tokens": response.prompt_tokens,
            "total_tokens": response.total_tokens,
        }
        
    def update(self,query_list, trajectory_list, reward_list, workflow_list, memory_list):  
        # vallina


        for memory in memory_list:
            for doc in self.documents:
                
                if doc.metadata.get("query") == memory:
                    doc.metadata["hit"] += 1
                    if reward_list[memory_list.index(memory)]:
                        doc.metadata["success"] += 1
                    break
        del_index = []
        for doc in self.documents:
            if doc.metadata.get("hit") >=3 and doc.metadata.get("success")/doc.metadata.get("hit") < 0.5:
                del_index.append(self.documents.index(doc))
        self.documents = [doc for i, doc in enumerate(self.documents) if i not in del_index]

        if self.update_policy == "vanilla":

            item_list = [{"source": "test", "query": query, "trajectory": trajectory} for query, trajectory in zip(query_list, trajectory_list)]
            new_documents = []
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_data = {executor.submit(self.process_trajectory_item, d): d for d in item_list}
                for future in tqdm(as_completed(future_to_data), desc="Building memory from trajectory", total=len(item_list)):
                    try:
                        doc = future.result()
                        if doc:  # Only add non-None results
                            new_documents.append(doc)
                    except Exception as e:
                        print(f"[ERROR] An error occurred while processing trajectory item: {e}")
            
            # Update documents list and save to disk
            self.documents.extend(new_documents)
            self._save_documents()

            print(f"[INFO] {len(new_documents)} new documents added.")

        elif self.update_policy == "validation":

            item_list = [{"source": "test", "query": query, "trajectory": trajectory} for query, trajectory, reward in zip(query_list, trajectory_list, reward_list) if reward]
            print(f"Filter out {len(item_list)}/{len(query_list)} items")
            new_documents = []
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_data = {executor.submit(self.process_trajectory_item, d): d for d in item_list}
                for future in tqdm(as_completed(future_to_data), desc="Building memory from trajectory", total=len(item_list)):
                    try:
                        doc = future.result()
                        if doc:  # Only add non-None results
                            new_documents.append(doc)
                    except Exception as e:
                        print(f"[ERROR] An error occurred while processing trajectory item: {e}")

            self.documents.extend(new_documents)
            self._save_documents()

        elif self.update_policy == "reflect":
            self.documents

            # filter reward true and false
            right_traj = []
            wrong_traj = []
            if len(workflow_list) == 0:
                workflow_list = [""]*len(query_list)
            for query, trajectory, reward, workflow in zip(query_list, trajectory_list, reward_list, workflow_list):
                if reward:
                    right_traj.append((query, trajectory, workflow, reward))
                else:
                    wrong_traj.append((query, trajectory, workflow, reward))
            new_documents = []
            with ThreadPoolExecutor(max_workers=16) as executor:
                future_to_data = {executor.submit(self.process_trajectory_item, {"source": "test", "query": query, "trajectory": trajectory}): (query, trajectory) for query, trajectory, _, _ in right_traj}
                for future in tqdm(as_completed(future_to_data), desc="Building memory from trajectory", total=len(right_traj)):
                    try:
                        doc = future.result()
                        if doc:  # Only add non-None results
                            new_documents.append(doc)
                    except Exception as e:
                        print(f"[ERROR] An error occurred while processing trajectory item: {e}")
            self.documents.extend(new_documents)

            with ThreadPoolExecutor(max_workers=1) as executor:
                future_to_data = {executor.submit(self.process_trajectory_item_reflect, trajectory, reward, workflow): (trajectory, reward, workflow) for _,trajectory, workflow, reward in wrong_traj}
                for future in tqdm(as_completed(future_to_data), desc="Reflecting memory", total=len(wrong_traj)):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] An error occurred while reflecting memory: {e}")
            self._save_documents()
            print(f"[INFO] {len(trajectory_list)} new documents added.")
        else:
            pass
       
       
if __name__=="__main__":    
    policy = {
        "build": "round",
        "retrieve": "query",
        "update": "",
    }
    travel_memory = Memory(is_cold_start=True, 
                           traj_file_path="ProcedureMem/test.json",
                           policy=policy,
                           retrieve_num=2,
                            memory_dir="test/memory")
    query = "Create a travel plan beginning in Oakland and heading to Tucson"

    workflow = travel_memory.retrieve(query).metadata.get("workflow")
    print(f"Retrieved workflow: {workflow}")
