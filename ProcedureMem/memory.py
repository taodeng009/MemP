import json
import os
import random
import numpy as np
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings
from langchain_core.documents import Document
from concurrent.futures import ThreadPoolExecutor, as_completed

from ProcedureMem.llm_api import get_llm_response, get_embedding_model
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

class Memory:
    def __init__(self, **kwargs):
        self.is_cold_start = kwargs.get("is_cold_start", False)
        self.policy = kwargs.get("policy", {})
        self.traj_file_path = kwargs.get("traj_file_path", None)
        self.retrieve_num = kwargs.get("retrieve_num", 10)
        self.memory_dir = kwargs.get("memory_dir", "memory")
        self.prompt_domain = kwargs.get("prompt_domain", "alfworld")

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


        os.makedirs(self.memory_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        os.makedirs(os.path.dirname(self.documents_path), exist_ok=True)

        # Initialize embedding model
        self.embedding = get_embedding_model()

        # Load document metadata
        

        if self.is_cold_start:
            self._cold_start()
        



    def _save_documents(self):
        """
        Save all current documents to disk.
        """
        with open(self.documents_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"page_content": d.page_content, "metadata": d.metadata} for d in self.documents],
                f, indent=2, ensure_ascii=False
            )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                build_prompt_manifest(self.build_policy),
                f,
                indent=2,
                ensure_ascii=False,
            )

        self.store = LocalFileStore(self.cache_dir)
        self.cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.embedding, self.store, namespace=self.embedding.model
        )

        
        # Create FAISS vector store from documents
        self.vector_store = FAISS.from_documents(self.documents, self.cached_embedder)

        self.doc_facts_embeddings = None
        #Compute and cache facts embeddings if needed
        if self.policy.get("retrieve") == "ave_fact":
            self.doc_facts_embeddings = load_facts_embedding_cache(self.facts_cache_path)
            if self.doc_facts_embeddings is None:
                print("[INFO] Computing facts embeddings for documents...")
                # Compute facts embeddings for all documents
                self.doc_facts_embeddings = compute_facts_embeddings(self.documents, self.embedding)
                save_facts_embedding_cache(self.facts_cache_path, self.doc_facts_embeddings)
            else:
                print(f"[INFO] Loaded facts embeddings from {self.facts_cache_path}")

    def process_trajectory_item(self, d):
        """
        Process a single trajectory item.
        This function includes logic for checking existence, building workflow, and appending new documents.
        """
        source = d.get("source")
        query = d.get("query").split("\n\n")[0]
        trajectory = d.get("trajectory")
        facts = d.get("facts", {})

        # A query built with a different prompt is not the same memory item.
        if any(
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
        content = json.dumps({"query": query, "workflow": workflow, "facts": facts})
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
            }
        )

        return doc

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
            events = get_llm_response(generate_events_from_trajectory_prompt(query, trajectory), is_string=False)
            workflow_ids = get_llm_response(generate_workflow_from_events_prompt(query, events), is_string=False)
            workflow = [events[wid - 1]['action'] for wid in workflow_ids]
        elif self.build_policy == "direct":
            workflow = get_llm_response(generate_workflow_from_trajectory_prompt(query, trajectory), is_string=True)
        
        return workflow

    def retrieve(self, key):
        """
        Retrieve from memory according to the specified policy.
        """
        retrieve_num = min(self.retrieve_num, len(self.documents))
        if self.retrieve_policy == "query":
            return self.vector_store.similarity_search_with_score(key, k=self.retrieve_num, score_threshold=0.5)

        elif self.retrieve_policy == "facts":
            key = str(key)
            return self.vector_store.similarity_search_with_score(key, k=self.retrieve_num, score_threshold=0.4)

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
