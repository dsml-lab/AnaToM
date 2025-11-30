import json
from collections import defaultdict
import re

def analyze_patterns_with_padding(stories_path, world_path, output_path):
    """
    パターン文字列の場所の区切り文字'/'が常に2つになるよう、
    空の場所を補って出力する最終版。
    """
    try:
        with open(stories_path, 'r', encoding='utf-8') as f:
            stories = json.load(f)
        with open(world_path, 'r', encoding='utf-8') as f:
            world_data = json.load(f)
    except FileNotFoundError as e:
        print(f"エラー: 入力ファイルが見つかりません。 ({e})")
        return

    WORLD_AGENTS = set(world_data.get("agents", []))
    WORLD_OBJECTS = set(world_data.get("objects", []))
    WORLD_CONTAINERS = set(world_data.get("containers", []))

    a3_o3_c3_stories = [s for s in stories if s.get("setting") == "A3_O3_C3"]

    categorized_patterns = defaultdict(lambda: defaultdict(int))
    category_counts = defaultdict(int)
    skipped_stories_report = defaultdict(list)

    for story in a3_o3_c3_stories:
        item_location, all_items = {}, set()
        for stmt in story["initial_state"]:
            match = re.match(r"(.+?)\s+(?:was|were)\s+in\s+the\s+(.+?)\.", stmt)
            if not match: continue
            items_raw, loc_name = match.group(1), match.group(2).strip()
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
        found_agents, found_objects, found_containers = 0, 0, 0
        for item in all_items:
            if item not in item_location: continue
            room = resolve_loc(item)
            if room is None: continue
            if item in WORLD_AGENTS:
                final_contents[room]["A"] += 1; found_agents += 1
            elif item in WORLD_CONTAINERS:
                final_contents[room]["C"] += 1; found_containers += 1
            elif item in WORLD_OBJECTS:
                final_contents[room]["O"] += 1; found_objects += 1

        if not (found_agents == 3 and found_objects == 3 and found_containers == 3):
            reason = f"設定と内容が不一致 (A:{found_agents}/3, O:{found_objects}/3, C:{found_containers}/3)"
            skipped_stories_report[reason].append(story['instance_index'])
            continue

        pattern_parts = []
        for room in sorted(final_contents.keys()):
            counts = final_contents[room]
            part = "A" * counts["A"] + "C" * counts["C"] + "O" * counts["O"]
            if part: pattern_parts.append(part)
        
        # ----- ▼▼▼ 今回の修正箇所 ▼▼▼ -----
        # 場所の数が3つに満たない場合、空文字列（""）で埋める
        num_to_pad = 3 - len(pattern_parts)
        if num_to_pad > 0:
            pattern_parts.extend([""] * num_to_pad)
        
        # ソートしてから結合することで、空の場所が先頭に来るようにする
        pattern_str = "/".join(sorted(pattern_parts))
        # ----- ▲▲▲ 今回の修正箇所 ▲▲▲ -----
        
        agent_dist = sorted([p.count('A') for p in pattern_parts if p.count('A') > 0], reverse=True)
        category = None
        if agent_dist == [3]: category = "AAA//"
        elif agent_dist == [2, 1]: category = "AA/A/"
        elif agent_dist == [1, 1, 1]: category = "A/A/A"
        
        if category:
            category_counts[category] += 1
            categorized_patterns[category][pattern_str] += 1
        else:
            skipped_stories_report["分類不能"].append(story['instance_index'])

    total_processed = sum(category_counts.values())
    total_skipped = len(a3_o3_c3_stories) - total_processed
    final_report = {
        reason: {"count": len(indices), "instance_indices": sorted(indices)}
        for reason, indices in skipped_stories_report.items()
    }

    results = {
        "analysis_summary": {
            "total_stories_in_setting": len(a3_o3_c3_stories),
            "processed_and_categorized": total_processed,
            "skipped_stories": total_skipped
        },
        "pattern_category_distribution": dict(sorted(category_counts.items())),
        "specific_pattern_distribution": {
            "A/A/A": dict(sorted(categorized_patterns["A/A/A"].items(), key=lambda item: item[1], reverse=True)),
            "AA/A/": dict(sorted(categorized_patterns["AA/A/"].items(), key=lambda item: item[1], reverse=True)),
            "AAA//": dict(sorted(categorized_patterns["AAA//"].items(), key=lambda item: item[1], reverse=True))
        },
        "skipped_stories_report": final_report
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"✅ パターン形式を修正した分析結果が {output_path} に保存されました。")


# --- プログラムの実行 ---
analyze_patterns_with_padding('stories.json', 'world.json', 'pattern.json')