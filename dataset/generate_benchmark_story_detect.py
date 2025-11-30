import json
import random
import re
from collections import defaultdict, Counter
from copy import deepcopy
from itertools import permutations

# ---------------------------------------------------------------------------
# 1. 設定とグローバル変数
# ---------------------------------------------------------------------------
WORLD_PATH = "world.json"
STORIES_JSON_PATH = "stories.json"
DISTRIBUTION_JSON_PATH = "distribution_analysis.json"
try:
    with open(WORLD_PATH, "r") as f: world = json.load(f)
except FileNotFoundError:
    print(f"エラー: {WORLD_PATH} が見つかりません。サンプルデータで続行します。")
    world = { "agents": [f"Agent-{i}" for i in range(10)], "objects": [f"Object-{i}" for i in range(10)], "containers": [f"Container-{i}" for i in range(10)], "locations": [f"Location-{i}" for i in range(10)], }

# ---------------------------------------------------------------------------
# 2. 状態表現とシミュレーションのためのクラス
# ---------------------------------------------------------------------------
class WorldState:
    def __init__(self, agent_locs, obj_locs, cont_locs):
        self.agent_locations = agent_locs
        self.object_locations = obj_locs
        self.container_locations = cont_locs
        self.locations = list(sorted(set(agent_locs.values()) | set(cont_locs.values())))
        self.belief_states = {}
        for agent in agent_locs.keys():
            self.belief_states[agent] = {obj: None for obj in obj_locs.keys()}
    
    def get_possible_moves(self):
        possible_actions = []
        agents, objects = list(self.agent_locations.keys()), list(self.object_locations.keys())
        for ag in agents:
            ag_loc = self.agent_locations.get(ag)
            if ag_loc == "unknown": continue
            for obj in objects:
                believed_cont = self.belief_states[ag].get(obj)
                actual_cont = self.object_locations.get(obj)
                if believed_cont != actual_cont: continue
                if self.container_locations.get(actual_cont) != ag_loc: continue
                possible_targets = [c for c, l in self.container_locations.items() if l == ag_loc and c != actual_cont]
                for target in possible_targets:
                    possible_actions.append({'type': 'move', 'agent': ag, 'object': obj, 'target': target})
        return possible_actions

    def get_possible_exits(self):
        possible_actions = []
        agents, locations = list(self.agent_locations.keys()), self.locations
        for ag in agents:
            current_loc = self.agent_locations.get(ag)
            other_locs = [loc for loc in locations if loc != current_loc]
            for new_loc in other_locs:
                possible_actions.append({'type': 'exit_enter', 'agent': ag, 'from': current_loc, 'to': new_loc})
        return possible_actions

# ---------------------------------------------------------------------------
# 3. 誤信念検知とイベント適用ロジック
# ---------------------------------------------------------------------------
def detect_false_belief(state: WorldState):
    fb_list = []
    for agent, beliefs in state.belief_states.items():
        for obj, believed_cont in beliefs.items():
            actual_cont = state.object_locations.get(obj)
            if believed_cont is not None and believed_cont != actual_cont:
                fb_list.append({"agent": agent, "object": obj, "believed_in": believed_cont, "actually_in": actual_cont})
    return fb_list

def apply_action_and_update_beliefs(state: WorldState, action: dict):
    new_state = deepcopy(state)
    agent = action['agent']
    if action['type'] == 'move':
        obj, target_cont = action['object'], action['target']
        new_state.object_locations[obj] = target_cont
        mover_loc = new_state.agent_locations[agent]
        for a, loc in new_state.agent_locations.items():
            if loc == mover_loc:
                new_state.belief_states[a][obj] = target_cont
    elif action['type'] == 'exit_enter':
        new_loc = action['to']
        new_state.agent_locations[agent] = new_loc
    return new_state

def analyze_belief_persistence(simulation_log):
    active_fbs, completed_fbs = {}, []
    for step_data in simulation_log:
        current_step = step_data['step']
        fbs_found_this_step = {(fb['agent'], fb['object']) for fb in step_data['false_beliefs_found']}
        for fb in step_data['false_beliefs_found']:
            key = (fb['agent'], fb['object'])
            if key not in active_fbs:
                active_fbs[key] = {"agent": fb['agent'], "object": fb['object'], "start_step": current_step, "end_step": None}
        resolved_keys = []
        for key, fb_info in list(active_fbs.items()):
            if key not in fbs_found_this_step:
                fb_info['end_step'] = current_step
                completed_fbs.append(fb_info)
                resolved_keys.append(key)
        for key in resolved_keys: del active_fbs[key]
    for key, fb_info in active_fbs.items():
        fb_info['end_step'] = "unresolved"
        completed_fbs.append(fb_info)
    for fb in completed_fbs:
        fb['duration_steps'] = (fb['end_step'] - fb['start_step']) if isinstance(fb['end_step'], int) else "N/A"
    return completed_fbs

def get_unique_permutations(partition):
    return sorted(list(set(permutations(partition))))

# ---------------------------------------------------------------------------
# 4. ストーリーとイベントの生成 (★修正箇所)
# ---------------------------------------------------------------------------
def create_story_with_fb_detection(structure, k_agents, k_objects, k_containers, k_locations=3, target_action_plan=None):
    if len(world["agents"]) < k_agents or len(world["objects"]) < k_objects or \
       len(world["containers"]) < k_containers or len(world["locations"]) < k_locations:
        return None
    agents, objects = random.sample(world["agents"], k_agents), random.sample(world["objects"], k_objects)
    containers, locations = random.sample(world["containers"], k_containers), random.sample(world["locations"], k_locations)
    agent_locs, obj_locs, cont_locs = {}, {}, {}
    la_partition, lc_partition = structure["la_partition"], structure["lc_partition"]
    la_perm = random.choice(get_unique_permutations(la_partition))
    lc_perm = random.choice(get_unique_permutations(lc_partition))
    la_iter, lc_iter = iter(agents), iter(containers)
    for i, count in enumerate(la_perm):
        for _ in range(count): agent_locs[next(la_iter)] = locations[i]
    for i, count in enumerate(lc_perm):
        for _ in range(count): cont_locs[next(lc_iter)] = locations[i]
    for obj in objects:
        obj_locs[obj] = random.choice(containers)

    initial_state_obj = WorldState(agent_locs, obj_locs, cont_locs)
    
    # ▼▼▼ 修正箇所: 初期状態の文章生成と信念設定のロジック ▼▼▼
    initial_sentences = []
    # 1. エージェントとコンテナの位置を記述
    for agent, loc in initial_state_obj.agent_locations.items():
        initial_sentences.append(f"{agent} was in the {loc}.")
    for container, loc in initial_state_obj.container_locations.items():
        initial_sentences.append(f"The {container} was in the {loc}.")

    # 2. オブジェクトの位置を静的に記述し、同時に初期信念を設定
    obj_by_cont = defaultdict(list)
    for obj, cont in initial_state_obj.object_locations.items(): 
        obj_by_cont[cont].append(obj)
        # このオブジェクトと同じ部屋にいるエージェントの信念を更新
        container_loc = initial_state_obj.container_locations.get(cont)
        for agent, agent_loc in initial_state_obj.agent_locations.items():
            if agent_loc == container_loc:
                initial_state_obj.belief_states[agent][obj] = cont
                
    for cont, objs in obj_by_cont.items():
        obj_str = " and ".join(sorted(objs))
        verb = "were" if len(objs) > 1 or any(s.endswith('s') for s in objs) else "was"
        initial_sentences.append(f"The {obj_str} {verb} in the {cont}.")
    
    # 3. 空の部屋について言及
    occupied_locations = set(initial_state_obj.agent_locations.values()) | set(initial_state_obj.container_locations.values())
    empty_locations = set(locations) - occupied_locations
    for loc in sorted(list(empty_locations)):
        initial_sentences.append(f"No one was in the {loc}.")

    current_state = initial_state_obj
    action_plan = target_action_plan
    simulation_log, has_false_belief_occurred = [], False
    
    for i, action_type in enumerate(action_plan):
        possible_actions = current_state.get_possible_moves() if action_type == 'move' else current_state.get_possible_exits()
        if not possible_actions: return None
        random.shuffle(possible_actions)
        best_action = None
        for action in possible_actions:
            is_safe_choice = True
            future_plan = action_plan[i+1:]
            if future_plan:
                tentative_next_state = apply_action_and_update_beliefs(current_state, action)
                if 'move' in future_plan and not tentative_next_state.get_possible_moves():
                    is_safe_choice = False
            if is_safe_choice:
                best_action = action
                break
        if not best_action: return None

        next_state = apply_action_and_update_beliefs(current_state, best_action)
        fbs_found = detect_false_belief(next_state)
        if fbs_found: has_false_belief_occurred = True
        event_sentence = f"{best_action['agent']} moved the {best_action['object']} to the {best_action['target']}." if best_action['type'] == 'move' else f"{best_action['agent']} exited {best_action['from']} and entered {best_action['to']}."
        simulation_log.append({"step": i + 1, "action_type": best_action['type'], "event": event_sentence, "false_beliefs_found": fbs_found})
        current_state = next_state

    if len(simulation_log) != 4: return None
    event_sentences = [log['event'] for log in simulation_log]
    full_story = initial_sentences + event_sentences
    return {"initial_state_sentences": initial_sentences, "simulation_log": simulation_log, "has_false_belief": has_false_belief_occurred, "action_sequence": [log['action_type'] for log in simulation_log], "full_story": full_story}

# ---------------------------------------------------------------------------
# 5. メイン実行部
# ---------------------------------------------------------------------------
def main():
    def get_partitions(n, k):
        if k == 0: return [[]] if n == 0 else [];
        if k == 1: return [[n]]
        res = []
        for i in range(n + 1):
            for sub in get_partitions(n - i, k - 1): res.append([i] + sub)
        return res
    def generate_valid_initial_states(k_agents, k_containers, k_locations=3):
        valid_structures, la_partitions, lc_partitions = [], get_partitions(k_agents, k_locations), get_partitions(k_containers, k_locations)
        for la in la_partitions:
            for lc in lc_partitions:
                if any(c > 1 for c in lc) and any(lc[i] > 1 and la[i] > 0 for i in range(k_locations)):
                    valid_structures.append({"la_partition": la, "lc_partition": lc})
        return valid_structures
    target_sequences = [['move', 'exit_enter', 'move', 'exit_enter'], ['move', 'exit_enter', 'exit_enter', 'move'], ['exit_enter', 'move', 'move', 'exit_enter'], ['exit_enter', 'move', 'exit_enter', 'move'], ['exit_enter', 'exit_enter', 'move', 'move']]
    final_stories, per_setting_distribution, instance_counter = [], defaultdict(Counter), 1
    settings_to_generate = [{"label": "A3_O3_C3", "k_a": 3, "k_o": 3, "k_c": 3}, {"label": "A4_O3_C3", "k_a": 4, "k_o": 3, "k_c": 3}, {"label": "A5_O3_C3", "k_a": 5, "k_o": 3, "k_c": 3}, {"label": "A3_O4_C3", "k_a": 3, "k_o": 4, "k_c": 3}, {"label": "A3_O5_C3", "k_a": 3, "k_o": 5, "k_c": 3}, {"label": "A3_O3_C4", "k_a": 3, "k_o": 3, "k_c": 4}, {"label": "A3_O3_C5", "k_a": 3, "k_o": 3, "k_c": 5}]
    for setting in settings_to_generate:
        setting_label, k_a, k_o, k_c = setting["label"], setting["k_a"], setting["k_o"], setting["k_c"]
        print(f"\n--- Processing setting: {setting_label} ---")
        valid_structures = generate_valid_initial_states(k_a, k_c)
        if not valid_structures: continue
        story_pool = []
        stories_per_sequence = 10000 // len(target_sequences)
        for seq in target_sequences:
            seq_name = " -> ".join(s.replace("_", "/") for s in seq)
            print(f"  Generating for sequence [{seq_name}]...")
            attempts, successful_stories, MAX_ATTEMPTS_PER_SEQ = 0, 0, 200000
            while successful_stories < stories_per_sequence and attempts < MAX_ATTEMPTS_PER_SEQ:
                attempts += 1
                story_data = create_story_with_fb_detection(random.choice(valid_structures), k_a, k_o, k_c, target_action_plan=seq)
                if story_data and story_data["has_false_belief"]:
                    belief_analysis = analyze_belief_persistence(story_data["simulation_log"])
                    if any(fb["end_step"] == "unresolved" for fb in belief_analysis):
                        story_data['false_belief_persistence'] = belief_analysis; story_pool.append(story_data); successful_stories += 1
        print(f"プールに {len(story_pool)} 件の「最後まで誤信念が残る」ストーリーを生成しました。")
        if len(story_pool) < 1000:
            print(f"警告: プール内のストーリーが1000件未満のため、{setting_label} をスキップします。")
            continue
        print(f"プールから1000件をランダムサンプリングします...")
        sampled_stories = random.sample(story_pool, 1000)
        for story_data in sampled_stories:
            sequence_tuple = tuple(story_data['action_sequence'])
            per_setting_distribution[setting_label][sequence_tuple] += 1
            final_stories.append({"instance_index": instance_counter, "setting": setting_label, "has_false_belief": story_data["has_false_belief"], "initial_state": story_data["initial_state_sentences"], "simulation_log": story_data["simulation_log"], "full_story": story_data["full_story"], "false_belief_persistence": story_data["false_belief_persistence"]})
            instance_counter += 1
    print(f"\n✍️  {len(final_stories)} 件のサンプリング結果を {STORIES_JSON_PATH} に保存しています...")
    with open(STORIES_JSON_PATH, "w", encoding='utf-8') as f: json.dump(final_stories, f, ensure_ascii=False, indent=2)
    print("✅ ストーリーの保存が完了しました。")
    print(f"✍️  イベント順序の分布を {DISTRIBUTION_JSON_PATH} に保存しています...")
    analysis_output = {}
    for setting, counter in sorted(per_setting_distribution.items()):
        total_for_setting = sum(counter.values())
        sequences = []
        for sequence, count in sorted(counter.items(), key=lambda item: item[1], reverse=True):
            percentage = (count / total_for_setting) * 100
            sequences.append({"sequence": " -> ".join(sequence), "count": count, "percentage": f"{percentage:.1f}%"})
        analysis_output[setting] = {"total_samples": total_for_setting, "distribution": sequences}
    with open(DISTRIBUTION_JSON_PATH, "w", encoding='utf-8') as f: json.dump(analysis_output, f, ensure_ascii=False, indent=2)
    print("✅ 分布データの保存が完了しました。")

if __name__ == "__main__":
    main()