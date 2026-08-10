import os
from litellm import completion

from alfworld.agents.environment import get_environment
from concurrent.futures import ThreadPoolExecutor, as_completed
from ProcedureMem.Alfworld.prompts import alfworld_system_prompt
from ProcedureMem.memory import Memory
from ProcedureMem.runtime_config import (
    DEFAULT_ALFWORLD_CONFIG,
    DEFAULT_EXAMPLES_PATH,
    DEFAULT_MEMORY_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_TRAJECTORY_PATH,
    configure_runtime,
    load_alfworld_config,
)
import copy
import argparse






def llm(prompt,stop=None, model=None):
    if isinstance(prompt, list):
        messages = prompt
    elif isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        raise ValueError(f'prompt must be a list or a string, but got {type(prompt)}')
    request_kwargs = {
        "model": model or os.environ["MODEL_NAME"],
        "messages": messages,
        "api_key": os.environ["OPENAI_API_KEY"],
        "num_retries": 10,
        "temperature": 1,
        "stop": stop,
    }
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base:
        request_kwargs["base_url"] = api_base
    response = completion(
        **request_kwargs
    )
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    return "Output Error"



def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob




def get_example(name,examples_list):
    prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
    for k, v in prefixes.items():
        if name.startswith(k):
            for example in examples_list:
                if example['task'] == v:
                    return example['example']
    assert False, f'{name} not found'


def alfworld_run_batch(obs:list=[],names:list=[],few_shot=True,max_steps=30,examples_list=[]):
    # Initialize messages list and active tasks
    messages_list = []
    active_tasks = list(range(len(obs)))  # Track active task indices
    



    for ob, name in zip(obs, names):
        messages = []
        messages.append({"role": "system", "content": alfworld_system_prompt})
        if few_shot:
            example = get_example(name,examples_list)
            example_copy = copy.deepcopy(example)
            example_copy[0]['content'] = "Here is an example of how to solve the task:\nExample:\n" + example_copy[0]['content']

            messages.extend(example_copy)
            messages.append({"role": "user", "content": "Now it's your turn.\n" + ob})
        else:
            messages.append({"role": "user", "content": ob})
        messages_list.append(messages)

    for i in range(max_steps):
        if not active_tasks:  # If no active tasks, break the loop
            break
        print(f'\033[91mActive tasks: {active_tasks}\033[0m')


        responses = {}
        with ThreadPoolExecutor(max_workers=len(active_tasks)) as executor:
            futures = {executor.submit(llm, messages_list[idx]): idx for idx in active_tasks}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    response = future.result()
                    print(f'\033[92mAgent {idx}: \n{response}\033[0m')
                    responses[idx] = response
                except Exception as e:
                    print(f'Error {idx}: {e}')
                    continue
        
        responses = dict(sorted(responses.items(), key=lambda x: x[0]))
        for idx, response in responses.items():
            messages_list[idx].append({"role": "assistant", "content": response})
        
        # Only process actions for active tasks
        action_list = [""] * len(obs)  # Initialize with empty actions
        for idx in active_tasks:
            if idx in responses:
                if 'Action: ' in responses[idx]:
                    action_list[idx] = responses[idx].split('Action: ')[-1].strip()
                else:
                    action_list[idx] = ""
        
        observation, reward, done, info = env.step(action_list)
        observation = [process_ob(ob) for ob in observation]
        print(f'\033[93mObservation: \n{observation}\033[0m')
        reward = info['won']

        # Update active tasks list
        new_active_tasks = []
        for idx in active_tasks:
            messages_list[idx].append({"role": "user", "content": f'Observation: {observation[idx]}'})
            if not done[idx]:
                new_active_tasks.append(idx)
        active_tasks = new_active_tasks
    
    return [{"messages": messages, "reward": reward, "name": name} for messages, reward, name in zip(messages_list, reward, names)]
        


def main(args):
    model_name = args.model
    output_path = DEFAULT_RESULTS_DIR / model_name / f'{args.split}_{args.exp_name}_few_shot_{args.few_shot}_memory_{args.use_memory}'


    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    #  memory init
    if args.use_memory:
        memory_config = {
            "is_cold_start": True,
            "policy": {
                "build": "direct",
                "retrieve": "query"
            },
            "traj_file_path": str(DEFAULT_TRAJECTORY_PATH),
            "retrieve_num": 10,
            "memory_dir": str(DEFAULT_MEMORY_DIR),
            "memory_size": 300,
            "prompt_domain": "alfworld",
        }
        Pro_Mem = Memory(**memory_config)

    # env init

    import json
    with DEFAULT_EXAMPLES_PATH.open('r', encoding='utf-8') as f:
        examples_list = json.load(f)


    import math
    from tqdm import tqdm
    finished_games = 0
    all_reward = 0



    for file in os.listdir(output_path):
        if file.endswith('.json'):
            finished_games += 1
            with open(output_path / file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                all_reward += result['reward']



    for idx in tqdm(range(math.ceil(num_games/env.batch_size))):

        ob_list, info = env.reset()
        if idx*env.batch_size + env.batch_size <=finished_games:
            continue
        ob_list = ['\n'.join(ob.split('\n\n')[1:]) for ob in ob_list]

        if args.use_memory:
            new_ob_list = []
            for ob in ob_list:
                query = ob.split('\n\n')[0]
                print(query)
                workflow = Pro_Mem.retrieve(query)
                workflow = [{"task_name": w.metadata.get('query'), "guidelines": w.metadata.get('workflow')} for w in workflow]
                workflow = json.dumps(workflow,indent=4,ensure_ascii=False)
                print(ob+'\n\n'+workflow)
                ob = ob + f'Here are some guidelines of how to solve the similar task:\n{workflow}\n'
                new_ob_list.append(ob)
            ob_list = new_ob_list


        print(ob)
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





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-4o')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--max_steps', type=int, default=30)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--few_shot', action='store_true')
    parser.add_argument('--use_memory', action='store_true')
    parser.add_argument('--alfworld-data', help='ALFWorld data root; defaults to ALFWORLD_DATA or ~/.cache/alfworld')
    parser.add_argument('--config', default=str(DEFAULT_ALFWORLD_CONFIG), help='ALFWorld YAML config')
    args = parser.parse_args()

    configure_runtime(
        model_name=args.model,
        alfworld_data=args.alfworld_data,
        require_llm=True,
        require_embedding=args.use_memory,
    )

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
