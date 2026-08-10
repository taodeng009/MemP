import os
from ProcedureMem.runtime_config import (
    DEFAULT_ALFWORLD_CONFIG,
    DEFAULT_EXAMPLES_PATH,
    DEFAULT_MEMORY_CONFIG,
    DEFAULT_RESULTS_DIR,
    configure_runtime,
    load_alfworld_config,
    load_memory_config,
)
from litellm import completion

from alfworld.agents.environment import get_environment
from ProcedureMem.Alfworld.prompts import alfworld_system_prompt
from ProcedureMem.alfworld_agent import resolve_litellm_model, run_alfworld_batch
from ProcedureMem.memory import Memory
import argparse






def llm(prompt,stop=None, model=None):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')
    api_base = os.getenv("OPENAI_API_BASE")
    model_name = resolve_litellm_model(model or os.environ["MODEL_NAME"], api_base)
    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "api_key": os.environ["OPENAI_API_KEY"],
        "num_retries": 10,
        "temperature": 1,
        "stop": stop,
    }
    if api_base:
        request_kwargs["api_base"] = api_base
    response = completion(**request_kwargs)
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    raise RuntimeError("LLM returned an empty response")



def alfworld_run_batch(obs=None, names=None, few_shot=True, max_steps=30, examples_list=None):
    return run_alfworld_batch(
        env=env,
        observations=obs or [],
        names=names or [],
        llm_fn=llm,
        system_prompt=alfworld_system_prompt,
        few_shot=few_shot,
        max_steps=max_steps,
        examples=examples_list or [],
    )
        


def main(args):
    model_name = args.model
    output_path = DEFAULT_RESULTS_DIR / model_name / f'{args.split}_{args.exp_name}_few_shot_{args.few_shot}_memory_{args.use_memory}'


    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    #  memory init
    if args.use_memory:
        memory_config = load_memory_config(args.memory_config)
        memory_config["memory_dir"] = f"{memory_config['memory_dir']}_{args.exp_name}"
        memory_config["build_model"] = args.memory_build_model
        Pro_Mem = Memory(**memory_config)

    # env init

    import json
    with DEFAULT_EXAMPLES_PATH.open('r', encoding='utf-8') as f:
        examples_list = json.load(f)


    import math
    from tqdm import tqdm
    finished_games = 0
    all_reward = 0


    # load finished games
    for file in os.listdir(output_path):
        if file.endswith('.json'):
            finished_games += 1
            with open(output_path / file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                all_reward += result['reward']



    batch_count = math.ceil(num_games / env.batch_size)
    if args.limit_batches is not None:
        batch_count = min(batch_count, args.limit_batches)
    for idx in tqdm(range(batch_count)):

        ob_list, info = env.reset()
        if idx*env.batch_size + env.batch_size <=finished_games:
            continue
        ob_list = ['\n'.join(ob.split('\n\n')[1:]) for ob in ob_list]
        query_list = [ob.split("\nYour task is to: ")[-1] for ob in ob_list]

        new_ob_list = []
        workflow_list = []
        memory_list = []
        if args.use_memory and len(Pro_Mem.documents) > 0:
            for ob in ob_list:
                query = ob.split("\nYour task is to: ")[-1]
                print(query)
                workflow = Pro_Mem.retrieve(query)
                workflow = [{"task_name": w[0].metadata.get("query"), "guidelines": w[0].metadata.get('workflow')} for w in workflow]
                memory_list.append(workflow[0]["task_name"])
                workflow_list.append(workflow[0]["guidelines"])
                workflow = json.dumps(workflow,indent=4,ensure_ascii=False)
                print(ob+'\n\n'+workflow)

                ob = ob + f'Here are some guidelines of how to solve the similar task:\n{workflow}\n'
                new_ob_list.append(ob)
            ob_list = new_ob_list
        name_list = ['/'.join(info['extra.gamefile'][i].split('/')[-3:-1]) for i in range(len(ob_list))]
        # get_prompt_list
        batch_results = alfworld_run_batch(obs=ob_list,names=name_list, few_shot=args.few_shot, max_steps=args.max_steps,examples_list=examples_list)


        for result in batch_results:
            all_reward += result['reward']
            finished_games += 1
        tqdm.write(f'Avg reward: {all_reward/finished_games}')

        

        for i, result in enumerate(batch_results):
            with open(output_path / f'idx_{idx*env.batch_size+i}.json', 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

        print(f'Finished {idx*env.batch_size+i+1} games')


        if args.use_memory and Pro_Mem.is_cold_start == False:
            trajectory_list = [result['messages'] for result in batch_results]
            reward_list = [result['reward'] for result in batch_results]
            Pro_Mem.update(query_list, trajectory_list, reward_list, workflow_list, memory_list)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Agent model; defaults to MODEL_NAME from the environment or .env')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--limit-batches', type=int, help='Run only the first N batches (for smoke tests)')
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--use_memory', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--memory-build-model', help='LLM used only for trajectory-to-workflow construction')
    parser.add_argument('--alfworld-data', help='ALFWorld data root; defaults to ALFWORLD_DATA or ~/.cache/alfworld')
    parser.add_argument('--config', default=str(DEFAULT_ALFWORLD_CONFIG), help='ALFWorld YAML config')
    parser.add_argument('--memory-config', default=str(DEFAULT_MEMORY_CONFIG), help='Memory YAML config')
    args = parser.parse_args()
    if args.limit_batches is not None and args.limit_batches < 1:
        parser.error('--limit-batches must be at least 1')

    settings = configure_runtime(
        model_name=args.model,
        alfworld_data=args.alfworld_data,
        require_llm=True,
        require_embedding=args.use_memory,
    )
    args.model = settings.model_name

    output_path = DEFAULT_RESULTS_DIR / args.model / f'{args.split}_{args.exp_name}_few_shot_{args.few_shot}_memory_{args.use_memory}'
    if args.overwrite and output_path.exists():
        for file in output_path.glob('*.json'):
            file.unlink()

    # env init
    config = load_alfworld_config(args.config)

    if args.split == 'dev':
        split = "eval_in_distribution"
    else:
        split = "eval_out_of_distribution"
    env = get_environment(config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=args.batch_size)
    num_games = len(env.gamefiles)
    print(num_games)
    main(args)
