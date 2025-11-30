import json
from collections import defaultdict
import re

def calculate_detailed_pattern_accuracy_v2(stories_path, world_path, eval_path, output_path):
    """
    詳細パターンと親パターン(AAA//など)の両方について、
    QAカテゴリごとの正答率を計算し、保存・表示する関数
    """
    print(f"Loading files...\n Stories: {stories_path}\n World: {world_path}\n Eval: {eval_path}")
    
    try:
        with open(stories_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        with open(world_path, 'r', encoding='utf-8') as f:
            world_data = json.load(f)
        with open(eval_path, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりません。 {e}")
        return

    # --- 1. ストーリーごとのパターン(詳細・親)を特定する ---
    print("Analyzing story patterns...")
    
    WORLD_AGENTS = set(world_data.get("agents", []))
    WORLD_OBJECTS = set(world_data.get("objects", []))
    WORLD_CONTAINERS = set(world_data.get("containers", []))

    # story_id -> {'specific': str, 'parent': str}
    story_info_map = {}
    target_stories = [s for s in stories if s.get("setting") == "A3_O3_C3"]

    for story in target_stories:
        item_location = {}
        all_items = set()

        for stmt in story["initial_state"]:
            match = re.match(r"(.+?)\s+(?:was|were)\s+in\s+the\s+(.+?)\.", stmt)
            if not match: continue
            
            items_raw = match.group(1)
            loc_name = match.group(2).strip()
            
            individual_items = [item.strip() for item in items_raw.replace("The ", "").split(" and ")]
            
            for item_name in individual_items:
                item_location[item_name] = loc_name
                all_items.add(item_name)

        cache = {}
        def resolve_loc(item):
            if item in cache: return cache[item]
            loc = item_location.get(item)
            final = resolve_loc(loc) if loc in all_items else loc
            cache[item] = final
            return final

        final_contents = defaultdict(lambda: {"A": 0, "C": 0, "O": 0})
        found_agents = 0
        for item in all_items:
            if item not in item_location: continue
            room = resolve_loc(item)
            if room is None: continue
            
            if item in WORLD_AGENTS:
                final_contents[room]["A"] += 1
                found_agents += 1
            elif item in WORLD_CONTAINERS:
                final_contents[room]["C"] += 1
            elif item in WORLD_OBJECTS:
                final_contents[room]["O"] += 1
        
        # エージェント数が3でない場合は対象外（念のため）
        if found_agents != 3:
            continue

        pattern_parts = []
        for room in sorted(final_contents.keys()):
            counts = final_contents[room]
            part = "A" * counts["A"] + "C" * counts["C"] + "O" * counts["O"]
            if part: pattern_parts.append(part)
        
        # --- 親パターンの判定 (エージェントの分布のみを見る) ---
        agent_counts = sorted([p.count('A') for p in pattern_parts if 'A' in p], reverse=True)
        parent_cat = "Unknown"
        if agent_counts == [3]: parent_cat = "AAA//"
        elif agent_counts == [2, 1]: parent_cat = "AA/A/"
        elif agent_counts == [1, 1, 1]: parent_cat = "A/A/A"

        # --- 詳細パターンの生成 (場所数パディング含む) ---
        num_to_pad = 3 - len(pattern_parts)
        if num_to_pad > 0:
            pattern_parts.extend([""] * num_to_pad)
            
        pattern_str = "/".join(sorted(pattern_parts))
        
        story_info_map[story['instance_index']] = {
            'specific': pattern_str,
            'parent': parent_cat
        }

    # --- 2. 評価結果を集計する ---
    print("Aggregating evaluation results...")
    
    # 集計用データ構造の初期化関数
    def init_stats():
        return {
            "story_ids": set(),
            "overall": {"correct": 0, "total": 0},
            "by_category": defaultdict(lambda: {"correct": 0, "total": 0})
        }

    specific_stats = defaultdict(init_stats)
    parent_stats = defaultdict(init_stats)

    for res in eval_results:
        idx = res.get("instance_index")
        
        if idx in story_info_map:
            info = story_info_map[idx]
            spec_pat = info['specific']
            parent_pat = info['parent']
            
            qa_cat = res.get("qa_category", "unknown")
            is_correct = res.get("is_correct", False)
            
            # 共通の更新処理
            def update_stats(stats_dict, key):
                stats_dict[key]["story_ids"].add(idx)
                stats_dict[key]["overall"]["total"] += 1
                stats_dict[key]["by_category"][qa_cat]["total"] += 1
                if is_correct:
                    stats_dict[key]["overall"]["correct"] += 1
                    stats_dict[key]["by_category"][qa_cat]["correct"] += 1

            # 詳細パターンと親パターンの両方で集計
            update_stats(specific_stats, spec_pat)
            update_stats(parent_stats, parent_pat)

    # --- 3. 正答率計算と整形 ---
    print("Calculating final metrics...")

    def process_stats_to_output(stats_dict):
        results = {}
        for key, data in stats_dict.items():
            # 総合
            ov_total = data["overall"]["total"]
            ov_correct = data["overall"]["correct"]
            ov_acc = ov_correct / ov_total if ov_total > 0 else 0
            
            # カテゴリ別
            cat_results = {}
            for cat, c_data in data["by_category"].items():
                c_total = c_data["total"]
                c_correct = c_data["correct"]
                c_acc = c_correct / c_total if c_total > 0 else 0
                cat_results[cat] = {
                    "correct": c_correct,
                    "total": c_total,
                    "accuracy": c_acc,
                    "accuracy_percent": f"{c_acc:.1%}"
                }
            
            results[key] = {
                "story_count": len(data["story_ids"]),
                "overall": {
                    "correct": ov_correct,
                    "total": ov_total,
                    "accuracy": ov_acc,
                    "accuracy_percent": f"{ov_acc:.1%}"
                },
                "by_category": dict(sorted(cat_results.items()))
            }
        # ストーリー数順でソート
        return dict(sorted(results.items(), key=lambda x: x[1]['story_count'], reverse=True))

    final_output = {
        "summary_by_parent_category": process_stats_to_output(parent_stats),
        "detailed_by_specific_pattern": process_stats_to_output(specific_stats)
    }

    # JSON保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"\n✅ 結果を {output_path} に保存しました。")
    
    # --- コンソール表示 ---
    print("\n=== 親カテゴリ別 正答率 ===")
    for cat, data in final_output["summary_by_parent_category"].items():
        print(f"[{cat}] (Stories: {data['story_count']})")
        print(f"  Overall: {data['overall']['accuracy_percent']}")
        print(f"  False Belief1: {data['by_category'].get('false_belief1_QA', {}).get('accuracy_percent', 'N/A')}")

    print("\n=== 詳細パターン別 正答率 (Top 3) ===")
    for i, (pat, data) in enumerate(list(final_output["detailed_by_specific_pattern"].items())[:3]):
        print(f"[{pat}] (Stories: {data['story_count']})")
        print(f"  Overall: {data['overall']['accuracy_percent']}")
        print(f"  False Belief1: {data['by_category'].get('false_belief1_QA', {}).get('accuracy_percent', 'N/A')}")

# --- 実行設定 ---
if __name__ == "__main__":
    # ファイルパス (適宜変更してください)
    STORIES_FILE = 'stories.json'
    WORLD_FILE = 'world.json'
    EVAL_FILE = './../../result_llama70BInstruct/result_llama70BInstruct_20251001_668h/evaluation_results.json'
    OUTPUT_FILE = 'pattern_accuracy_llama70B.json'

    calculate_detailed_pattern_accuracy_v2(STORIES_FILE, WORLD_FILE, EVAL_FILE, OUTPUT_FILE)