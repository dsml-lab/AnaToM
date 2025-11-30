import json
import random
import re
from copy import deepcopy
from pathlib import Path
from collections import defaultdict, Counter

# ---------------------------------------------------------------------------
# 1. 設定とグローバル変数
# ---------------------------------------------------------------------------
STORIES_IN_PATH = Path("stories.json")
QA_OUT_PATH = Path("qa_sets.json")
WORLD_PATH = Path("world.json")

# ---------------------------------------------------------------------------
# 2. ユーティリティ関数
# ---------------------------------------------------------------------------
def is_plural(noun: str) -> bool:
    return noun.endswith("s") and not noun.endswith("ss")

def verb_agree(noun: str, singular: str, plural: str) -> str:
    return plural if is_plural(noun) else singular

# ---------------------------------------------------------------------------
# 3. 状態管理クラス
# ---------------------------------------------------------------------------
class RealityState:
    def __init__(self, agent_locs, obj_locs, cont_locs):
        self.agent_locs, self.obj_locs, self.cont_locs = agent_locs, obj_locs, cont_locs
    def apply(self, event: str):
        toks = event.split()
        if not toks: return
        try:
            if "moved" in toks:
                obj_index = toks.index("the") + 1
                self.obj_locs[toks[obj_index]] = toks[-1].rstrip(".")
            elif "entered" in toks:
                self.agent_locs[toks[0]] = toks[toks.index("entered") + 1].rstrip(".")
        except (ValueError, IndexError): pass

class BeliefState:
    def __init__(self, agent_locs, obj_locs, cont_locs):
        self.agent_locations, self.object_locations, self.container_locations = agent_locs, obj_locs, cont_locs
        self.belief_states = {ag: {obj: None for obj in obj_locs} for ag in agent_locs}

# ---------------------------------------------------------------------------
# 4. パーサーとシミュレーター (★修正箇所)
# ---------------------------------------------------------------------------
def parse_initial_state(sentences: list, locations: set):
    """初期状態の文章を解析して、各要素の位置辞書を作成する"""
    agent_locs, cont_locs, obj_locs = {}, {}, {}
    patterns = {
        "agent": re.compile(r"^([\w-]+) was in the ([\w_-]+)\.$"),
        "container_or_obj": re.compile(r"^The ([\w\sand-]+) (?:was|were) in the ([\w_-]+)\.$"),
    }
    
    # Pass 1: まずコンテナを特定する
    for s in sentences:
        m = patterns["container_or_obj"].match(s)
        if m and m[2] in locations:
            # "The [X] was in the [LOCATION]." -> Xはコンテナ
            cont_locs[m[1]] = m[2]

    # Pass 2: エージェントとオブジェクトを特定
    for s in sentences:
        m_ag = patterns["agent"].match(s)
        if m_ag:
            agent_locs[m_ag[1]] = m_ag[2]
            continue
        
        m_obj = patterns["container_or_obj"].match(s)
        if m_obj and m_obj[2] in cont_locs:
            # "The [Y] was in the [CONTAINER]." -> Yはオブジェクト
            obj_names = [o.strip() for o in m_obj[1].split(" and ")]
            for obj in obj_names:
                obj_locs[obj] = m_obj[2]

    return agent_locs, obj_locs, cont_locs

def apply_event_for_belief(state: BeliefState, event: str):
    new_state = deepcopy(state)
    toks = event.split()
    if not toks: return new_state
    try:
        if "moved" in toks:
            agent, target_cont = toks[0], toks[-1].rstrip('.')
            the_index, prep_index = toks.index("the"), toks.index("to")
            obj_str = " ".join(toks[the_index + 1 : prep_index])
            objects = [o.strip() for o in obj_str.split(" and ")]
            for obj in objects: new_state.object_locations[obj] = target_cont
            mover_loc = new_state.agent_locations.get(agent)
            if mover_loc:
                for ag, loc in new_state.agent_locations.items():
                    if loc == mover_loc:
                        for obj in objects: new_state.belief_states[ag][obj] = target_cont
        elif "entered" in toks:
            agent, new_loc = toks[0], toks[toks.index("entered") + 1].rstrip('.')
            new_state.agent_locations[agent] = new_loc
    except (ValueError, IndexError): pass
    return new_state

# ---------------------------------------------------------------------------
# 5. 課題生成のメインロジック
# ---------------------------------------------------------------------------
def build_qa_for_story(story: dict, locations: set):
    event_sentences = [log['event'] for log in story.get("simulation_log", [])]
    agent_locs, obj_locs, cont_locs = parse_initial_state(story["initial_state"], locations)

    if not obj_locs or not agent_locs:
        return None # 解析に失敗したストーリーはスキップ

    initial_reality = RealityState(deepcopy(agent_locs), deepcopy(obj_locs), deepcopy(cont_locs))
    final_reality = RealityState(deepcopy(agent_locs), deepcopy(obj_locs), deepcopy(cont_locs))
    for ev in event_sentences:
        final_reality.apply(ev)

    initial_belief_state = BeliefState(agent_locs, obj_locs, cont_locs)
    for agent, agent_loc in initial_belief_state.agent_locations.items():
        for obj, obj_cont in initial_belief_state.object_locations.items():
            if initial_belief_state.container_locations.get(obj_cont) == agent_loc:
                initial_belief_state.belief_states[agent][obj] = obj_cont
    snapshots = [initial_belief_state]
    for ev in event_sentences:
        snapshots.append(apply_event_for_belief(snapshots[-1], ev))
        
    final_belief_state = snapshots[-1]
    agents, objects = sorted(agent_locs.keys()), sorted(obj_locs.keys())
    memory_qa, reality_qa, true1_qa, false1_qa, true2_qa, false2_qa = [], [], [], [], [], []
    
    for obj in objects:
        verb_was, verb_is = verb_agree(obj, "was", "were"), verb_agree(obj, "is", "are")
        memory_qa.append((f'Where {verb_was} the {obj} at the beginning?', initial_reality.obj_locs.get(obj, "unknown")))
        reality_qa.append((f'Where {verb_is} the {obj} now?', final_reality.obj_locs.get(obj, "unknown")))
    for ag in agents:
        for obj in objects:
            believed, actual = final_belief_state.belief_states[ag].get(obj), final_reality.obj_locs.get(obj)
            if believed is None: continue
            q = f'Where does {ag} think the {obj} {verb_agree(obj, "is", "are")}?'
            (true1_qa if believed == actual else false1_qa).append((q, believed))
    last_seen = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: -1)))
    room2agents0 = defaultdict(list)
    for ag, loc in snapshots[0].agent_locations.items(): room2agents0[loc].append(ag)
    for room, agents_in_room in room2agents0.items():
        objs_in_room = {o for o, c in snapshots[0].object_locations.items() if snapshots[0].container_locations.get(c) == room}
        for a1 in agents_in_room:
            for a2 in agents_in_room:
                if a1 != a2:
                    for obj in objs_in_room: last_seen[a1][a2][obj] = 0
    for step, ev in enumerate(event_sentences, 1):
        if "moved" in ev:
            try:
                toks = ev.split()
                moved_obj, new_cont = toks[toks.index("the") + 1], toks[-1].rstrip(".")
                move_loc = snapshots[step].container_locations.get(new_cont)
                if move_loc:
                    agents_in_room = [ag for ag, loc in snapshots[step].agent_locations.items() if loc == move_loc]
                    for a1 in agents_in_room:
                        for a2 in agents_in_room:
                            if a1 != a2: last_seen[a1][a2][moved_obj] = step
            except (ValueError, IndexError): continue
    for obj in objects:
        actual = final_reality.obj_locs.get(obj)
        for ag1 in agents:
            for ag2 in agents:
                if ag1 == ag2: continue
                last_seen_step = last_seen[ag1][ag2].get(obj, -1)
                if last_seen_step == -1: continue
                ag1s_guess, ag2s_final_belief = snapshots[last_seen_step].belief_states[ag2].get(obj), final_belief_state.belief_states[ag2].get(obj)
                if ag2s_final_belief is None or ag1s_guess is None: continue
                q = f'Where does {ag1} think that {ag2} thinks the {obj} {verb_agree(obj, "is", "are")}?'
                a = ag1s_guess
                if ag1s_guess == ag2s_final_belief: true2_qa.append((q, a))
                else: false2_qa.append((q, a))
    def sample_one(qa_list):
        if not qa_list: return []
        q, a = random.choice(qa_list)
        return [{"question": q, "answer": a}]
    return {"instance_index": story.get("instance_index"), "setting": story.get("setting"), "full_story": story.get("full_story"), "memory_QA": sample_one(memory_qa), "reality_QA": sample_one(reality_qa), "true_belief1_QA": sample_one(true1_qa), "false_belief1_QA": sample_one(false1_qa), "true_belief2_QA": sample_one(true2_qa), "false_belief2_QA": sample_one(false2_qa)}

# ---------------------------------------------------------------------------
# 6. メイン実行部
# ---------------------------------------------------------------------------
def main():
    try:
        stories = json.loads(STORIES_IN_PATH.read_text(encoding="utf-8"))
        world_data = json.loads(WORLD_PATH.read_text(encoding="utf-8"))
        locations = set(world_data.get("locations", []))
    except FileNotFoundError as e:
        print(f"エラー: 入力ファイルが見つかりません。 ({e})")
        return
        
    qa_sets, qa_counts = [], Counter()
    qa_categories = ["memory_QA", "reality_QA", "true_belief1_QA", "false_belief1_QA", "true_belief2_QA", "false_belief2_QA"]
    for st in stories:
        qa_set = build_qa_for_story(st, locations)
        if qa_set: # Noneでない場合のみ追加
            qa_sets.append(qa_set)
            for category in qa_categories:
                if qa_set.get(category): qa_counts[category] += 1
    
    QA_OUT_PATH.write_text(json.dumps(qa_sets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ {len(qa_sets)}件のストーリーから課題を生成し、{QA_OUT_PATH} に保存しました。")
    print("\n--- 課題生成サマリー ---")
    print(f"処理したストーリーの総数: {len(stories)}件")
    print("各課題タイプについて生成された設問数:")
    for category, count in sorted(qa_counts.items()):
        print(f"  - {category:<20}: {count}問")

if __name__ == "__main__":
    main()