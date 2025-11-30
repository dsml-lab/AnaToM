import json
import re
import string
import os
from collections import defaultdict
from openai import OpenAI
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv # ★ 1. .envファイルを読み込むライブラリをインポート

# ---------------------------------------------------------------------------
# 1. 設定とグローバル変数
# ---------------------------------------------------------------------------

MODEL_NAME = "gpt-4.1-mini"
QA_SETS_PATH = Path("qa_sets.json")
RESULTS_PATH = Path(f"evaluation_results_{MODEL_NAME.replace('/', '_')}.json")
SUMMARY_PATH = Path(f"evaluation_summary_{MODEL_NAME.replace('/', '_')}.json")

# ---------------------------------------------------------------------------
# 2. モデルの準備 (★修正箇所)
# ---------------------------------------------------------------------------
def setup_llm_client():
    """環境変数または.envファイルからAPIキーを読み込み、OpenAIクライアントをセットアップする"""
    print(f"Setting up client for model: {MODEL_NAME}...")
    try:
        # ▼▼▼ 修正: .envファイルを読み込む処理を追加 ▼▼▼
        load_dotenv()
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("エラー: OPENAI_API_KEYが見つかりません。")
            print(".envファイルにキーを設定したか、または環境変数として設定されているか確認してください。")
            exit()
            
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error setting up OpenAI client: {e}")
        exit()

# ---------------------------------------------------------------------------
# 3. LLMとの対話と評価
# ---------------------------------------------------------------------------
def ask_llm(prompt: str, client: OpenAI):
    """プロンプトを整形し、OpenAIのモデルに問い合わせて回答を抽出する"""
    messages = [
        {"role": "system", "content": "You are an expert in reading comprehension. Answer the following question based ONLY on the text provided in the story. Provide only the answer, without any introductory phrases or explanations."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=50,
            temperature=0.0,
            top_p=1.0,
        )
        answer = response.choices[0].message.content.strip()
        return answer
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return "ERROR: API call failed"

def are_answers_equivalent(llm_answer: str, ground_truth: str) -> bool:
    """LLMの回答と正解を比較し、正誤を判定する"""
    if not llm_answer:
        return False
    
    llm_clean = llm_answer.strip().lower().rstrip(string.punctuation)
    gt_clean = ground_truth.strip().lower().rstrip(string.punctuation)

    if gt_clean in ["no one", "empty"]:
        return gt_clean in llm_clean

    negation_words = ["not", "never", "no "]
    if any(word in llm_clean for word in negation_words):
        return False

    pattern = r'\b' + re.escape(gt_clean) + r'\b'
    return bool(re.search(pattern, llm_clean))

# ---------------------------------------------------------------------------
# 4. メイン処理
# ---------------------------------------------------------------------------
def main():
    llm_client = setup_llm_client()

    print(f"Loading data from {QA_SETS_PATH}...")
    try:
        qa_sets = json.loads(QA_SETS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"エラー: 入力ファイル {QA_SETS_PATH} が見つかりません。")
        return

    tasks_to_evaluate = []
    for qa_set in qa_sets:
        story_text = "\n".join(qa_set['full_story'])
        for qa_category, qa_list in qa_set.items():
            if not qa_category.endswith("_QA") or not qa_list:
                continue
            for qa_pair in qa_list:
                tasks_to_evaluate.append({
                    "instance_index": qa_set["instance_index"],
                    "qa_category": qa_category,
                    "full_story_text": story_text,
                    "question": qa_pair['question'],
                    "ground_truth_answer": qa_pair['answer'],
                    "setting": qa_set.get("setting", "unknown"),
                })

    evaluation_details = []
    print(f"Starting evaluation of {len(tasks_to_evaluate)} questions with {MODEL_NAME}...")
    
    for task in tqdm(tasks_to_evaluate, desc="Evaluating Questions"):
        prompt = (
            "Please read the following story and answer the subsequent question.\n\n"
            "--- STORY ---\n"
            f"{task['full_story_text']}\n"
            "--- END OF STORY ---\n\n"
            f"Question: {task['question']}"
        )
        llm_answer = ask_llm(prompt, llm_client)
        is_correct = are_answers_equivalent(llm_answer, task['ground_truth_answer'])
        
        task['llm_answer'] = llm_answer
        task['is_correct'] = is_correct
        evaluation_details.append(task)

    print("\n--- Evaluation Summary ---")
    summary_data = {"overall_accuracy": {}, "accuracy_by_category": {}}
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for detail in evaluation_details:
        cat = detail["qa_category"]
        category_stats[cat]["total"] += 1
        if detail["is_correct"]:
            category_stats[cat]["correct"] += 1
            
    overall_correct = sum(d['correct'] for d in category_stats.values())
    overall_total = sum(d['total'] for d in category_stats.values())
    
    if overall_total > 0:
        summary_data["overall_accuracy"] = {
            "accuracy": overall_correct / overall_total,
            "correct": overall_correct,
            "total": overall_total
        }
        print(f"Overall Accuracy: {summary_data['overall_accuracy']['accuracy']:.2%}")
    
    for category, data in sorted(category_stats.items()):
        if data['total'] > 0:
            accuracy = data['correct'] / data['total']
            summary_data["accuracy_by_category"][category] = {
                "accuracy": accuracy,
                "correct": data['correct'],
                "total": data['total']
            }
            print(f"  - {category:<20}: {accuracy:.2%} ({data['correct']}/{data['total']})")

    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(evaluation_details, f, ensure_ascii=False, indent=2)
    print(f"\nFull raw results saved to {RESULTS_PATH}")
    
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    print(f"Summary saved to {SUMMARY_PATH}")

if __name__ == "__main__":
    main()