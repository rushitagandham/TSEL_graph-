import os
import json
from textwrap import dedent

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq

# -------------------------------------------------------------
# 全体設定
# -------------------------------------------------------------

# 日本語フォント（Windows）
matplotlib.rcParams["font.family"] = "Noto Serif CJK JP"

st.set_page_config(page_title="学習者レポートデモ", layout="wide")




# Groq クライアント（AI インサイト用）
# ※ APIキーは環境変数などで安全に管理してください
GROQ_API_KEY = None
if GROQ_API_KEY is None:
    st.warning("環境変数『GROQ_API_KEY』が設定されていないため、AIインサイト機能は利用できません。")
    groq_client = None
else:
    groq_client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama-3.3-70b-versatile"

# -------------------------------------------------------------
# データ読み込み
# -------------------------------------------------------------

@st.cache_data
def load_students(path: str = "students.json"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    students = {s["student_id"]: s for s in data["students"]}
    return students

students = load_students()

# -------------------------------------------------------------
# ルールベース分析（AI への入力用）
# -------------------------------------------------------------

def analyze_student(student: dict) -> dict:
    attendance = student["attendance"]
    homework = student["homework"]
    tests = student["tests"]
    progress = student["progress"]

    strengths, weaknesses, notes = [], [], []

    # 出席
    if attendance["pct"] >= 90:
        strengths.append("出席率がとても良好です。")
    elif attendance["pct"] >= 80:
        notes.append("出席率はおおむね良好ですが、もう少し安定するとより安心です。")
    else:
        weaknesses.append("グループ平均と比べて出席率が低めです。")

    # 宿題
    if homework["homework_completion_pct"] >= 90:
        strengths.append("宿題の実施率が高く、よく取り組めています。")
    elif homework["homework_completion_pct"] >= 70:
        notes.append("宿題の実施率はまずまずですが、もう一歩安定させたいところです。")
    else:
        weaknesses.append("宿題の実施率が低く、課題への取り組みが不足しています。")

    # スキル別テスト
    for skill, score in tests["by_skill"].items():
        if score >= 80:
            strengths.append(f"{skill.capitalize()} のテストスコアが高く（{score}%）、よく理解できています。")
        elif score < 60:
            weaknesses.append(f"{skill.capitalize()} のテストスコアが低め（{score}%）で、追加練習が必要です。")

    # 全体進捗
    kanna_pct = 100 * progress["kanna_completed"] / progress["kanna_total"]
    alpha_pct = 100 * progress["alpha_completed"] / progress["alpha_total"]
    avg_progress = (kanna_pct + alpha_pct) / 2

    if avg_progress >= 90:
        strengths.append("全体の学習進捗は予定どおり、またはそれ以上のペースで進んでいます。")
    elif avg_progress < 70:
        weaknesses.append("全体の学習進捗が想定ペースより遅れている状態です。")

    overall_status = "on_track"
    if any("低" in w or "遅れ" in w or "不足" in w for w in weaknesses):
        overall_status = "at_risk"
    elif weaknesses:
        overall_status = "needs_attention"

    return {
        "overall_status": overall_status,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "notes": notes,
        "avg_progress_pct": avg_progress
    }

# -------------------------------------------------------------
# Groq Llama 3.3 による AI インサイト生成
# -------------------------------------------------------------

def generate_student_insights_llama(
    student: dict,
    analysis: dict,
    language: str = "ja",
    max_tokens: int = 512
) -> str:
    if groq_client is None:
        return "（GROQ_API_KEY が設定されていないため、AIインサイトは利用できません。）"

    if language == "en":
        lang_instruction = "レポートは英語で作成してください。"
    elif language == "ja":
        lang_instruction = "レポートは日本語で作成してください。"
    else:
        lang_instruction = (
            "まず英語で、次に同じ内容を日本語で繰り返して書いてください。"
        )

    system_instructions = dedent(f"""
        あなたは日本語教育の講師として、学習者の成績データや行動データから
        「講師コメント」を作成する専門アシスタントです。

        出力形式は必ず次の3区分で作成してください：

        【成長】
        - 学習者ができるようになったこと
        - 伸びている能力
        - 良い変化

        【要改善】
        - まだ不安定な部分
        - 追加の練習が必要な領域

        【改善】
        - 短期的に取り組むべき行動
        - 次の1〜2週間で実施すると効果が高い学習方法

        ルール：
        - 数値を作らない（与えられたデータの範囲で記述する）
        - 文章は自然で、実際に講師が書くコメントのように書く
        - 過度に厳しくせず、前向きで丁寧な語尾にする
        - 各区分に2〜3文ずつ書く
        - 箇条書き記号（「-」など）は使わず、自然な文章として書く
        - 不必要な分析や結論は書かない

        - {lang_instruction}
    """).strip()

    profile_min = {
        "student_id": student["student_id"],
        "student_name": student["student_name"],
        "group_name": student["group_name"],
        "level": student["level"],
        "period": student["period"],
    }

    profile_json = json.dumps(profile_min, ensure_ascii=False, indent=2)
    tests_json = json.dumps(student["tests"], ensure_ascii=False, indent=2)
    attendance_json = json.dumps(student["attendance"], ensure_ascii=False, indent=2)
    homework_json = json.dumps(student["homework"], ensure_ascii=False, indent=2)
    progress_json = json.dumps(student["progress"], ensure_ascii=False, indent=2)
    analysis_json = json.dumps(analysis, ensure_ascii=False, indent=2)

    user_prompt = dedent(f"""
        学習者プロフィール:
        {profile_json}

        テストデータ:
        {tests_json}

        出席データ:
        {attendance_json}

        宿題データ:
        {homework_json}

        進捗データ:
        {progress_json}

        ルールベース分析結果:
        {analysis_json}

        上記の情報のみを用いて、次の3つの見出しで講師コメントを書いてください。

        【成長】
        学習者ができるようになったこと・伸びている点・良い変化など

        【要改善】
        まだ不安定な部分・課題として意識してほしい点など

        【改善】
        次の1〜2週間で取り組むと効果が高い具体的な学習行動やアドバイス
    """).strip()

    completion = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.4,
        top_p=0.9,
    )

    return completion.choices[0].message.content.strip()

# -------------------------------------------------------------
# グラフ描画用ヘルパー
# -------------------------------------------------------------

def graph1_current_mastery(student: dict):
    cm = student["current_mastery"]
    months = cm["months"]
    actual = cm["scores"]
    plan = cm["plan"]
    pass_line = cm.get("pass_line", 60)
    target_line = cm.get("target_line", 80)

    actual = [np.nan if v is None else v for v in actual]

    fig, ax = plt.subplots(figsize=(3,3))
    bar_width = 0.25
    x = np.arange(len(months))

    ax.bar(x - bar_width/2, actual, width=bar_width, label="実績", color="#f6a37a")
    ax.bar(x + bar_width/2, plan,   width=bar_width, label="計画", color="#1f77b4")

    ax.axhline(pass_line, linestyle="--", color="orange")
    ax.text(len(months)-0.3, pass_line+1, "合格最低ライン", ha="right", va="bottom")

    ax.axhline(target_line, linestyle="--", color="blue")
    ax.text(len(months)-0.3, target_line+1, "目標ライン", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(months)
    ax.set_ylim(0, 90)
    ax.set_ylabel("正答率（％）")
    ax.set_title(" 現在の習得度（月末テストより算出）",fontsize =8, pad=10)
    ax.legend()

    st.pyplot(fig)

def graph2_attendance_radar(student: dict):
    att = student["attendance"]
    labels = [
        "① 授業の出席率",
        "② 遅刻の有無",
        "③ 授業態度"
    ]
    values = [
        att["attendance_score"],
        att["lateness_score"],
        att["attitude_score"],
    ]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels)+1)

    fig, ax = plt.subplots(figsize=(2.5,2.5), subplot_kw=dict(polar=True))
    ax.set_rgrids([0, 1, 2, 3, 4, 5], angle=90)

    ax.plot(angles, values, linewidth=2, color="#1f77b4")
    ax.fill(angles, values, color="#1f77b4", alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(" 出席・受講態度", pad=20)

    st.pyplot(fig)

def graph3_learning_time(student: dict):

    stime = student["study_time"]
    # 元データをまとめておく
    items = [
        ("中級カンナが行く", stime["kanna_hours"],       stime["group_kanna_hours"]),
        ("中級日本語α",     stime["alpha_hours"],       stime["group_alpha_hours"]),
    ]

    labels = []
    bars = []
    for course, student_hours, group_hours in items:
        # 2行表示用に改行を入れる（コー名 + （本人／グループ平均））
        labels.append(f"{course}\n（本人）")
        bars.append(student_hours)
        labels.append(f"{course}\n（グループ平均）")
        bars.append(group_hours)

    y_pos = np.arange(len(labels))
    colors = ["#1f77b4" if i % 2 == 0 else "#9bbad1" for i in range(len(bars))]

    # ★ 右側の棒グラフと同じくらいのサイズに & 棒を太めに
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.barh(y_pos, bars, color=colors, height=0.9)  # height を大きめにして太くする

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("学習時間（時間）", fontsize=12)
    ax.set_title("学習時間比較",fontsize =14, pad=10)

    max_val = max(bars) if bars else 0
    ax.set_xlim(0, max_val * 1.15)  # 右側に少し余白

    # ★ 各バーの値を表示（ツールチップ風ラベル）
    for i, v in enumerate(bars):
        ax.text(
            v + max_val * 0.02,  # 棒のすぐ右側
            i,
            f"{v:.1f}h",
            va="center",
            ha="left",
            fontsize=11,
        )

    plt.tight_layout()
    st.pyplot(fig)



def draw_percentage_circle(title: str, percent: float, color: str):
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.pie(
        [percent, 100 - percent],
        colors=[color, "white"],
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.9},
    )

    ax.plot([0, 0], [0, 1.3], color="white", linewidth=3, transform=ax.transAxes)
    ax.text(0.5,-0.0, f"{percent:.0f}%", ha="center", va="center",
            fontsize=15, transform=ax.transAxes)
    ax.set_title(title, fontsize=14, pad=2)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig


def graph4_homework(student: dict):
    hw = student["homework"]


    col1, col2 = st.columns(2)
    with col1:
        fig1 = draw_percentage_circle("① ノート提出率", hw["notebook_submission_pct"], "#4CAF50")
        st.pyplot(fig1)
    with col2:
        fig2 = draw_percentage_circle("② 宿題実施率", hw["homework_completion_pct"], "#4CAF50")
        st.pyplot(fig2)

def graph5_test_triangle(student: dict):
    t = student["tests"]
    labels = [
        "① カンナテスト\n（10点）",
        "② αテスト\n（10点）",
        "③ 月末テスト\n（10点）"
    ]
    values = [t["kanna_score"], t["alpha_score"], t["monthly_score"]]
    values += values[:1]
    angles = np.linspace(0, 2*np.pi, len(labels)+1)

    fig, ax = plt.subplots(figsize=(2,2), subplot_kw=dict(polar=True))
    ax.set_rgrids([0, 2, 4, 6, 8, 10], angle=90)

    ax.plot(angles, values, linewidth=2, color="#1f77b4")
    ax.fill(angles, values, color="#1f77b4", alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 10)
    ax.set_title(" 理解・テスト", pad=20)
    ax.grid(color="gray", alpha=0.3)

    st.pyplot(fig)

def draw_progress_circle(label: str, total_lessons: int, completed: int):
    percent = 100 * completed / total_lessons
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    ax.pie(
        [percent, 100 - percent],
        colors=["#F28A2E", "white"],
        startangle=90,
        counterclock=False,
        wedgeprops={'width': 0.9},
    )

    ax.plot([0, 0], [0, 1.3], color="white", linewidth=3, transform=ax.transAxes)

    ax.text(0.5, 1,
            f"{label}   全lesson{total_lessons}",
            ha="center", va="center", fontsize=12, transform=ax.transAxes)

    ax.text(0.5,-0.05, f"{percent:.0f}%", ha="center", va="center",
            fontsize=15, fontweight="bold", transform=ax.transAxes)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig

def graph6_progress(student: dict):
    prog = student["progress"]
    

    col1, col2 = st.columns(2)
    with col1:
        fig1 = draw_progress_circle(
            "カンナ",
            prog["kanna_total"],
            prog["kanna_completed"],
        )
        st.pyplot(fig1)
    with col2:
        fig2 = draw_progress_circle(
            "α",
            prog["alpha_total"],
            prog["alpha_completed"],
        )
        st.pyplot(fig2)

# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------

# ============================================
# 🔹 ページタイトル & サイドバー（学習者情報）
# ============================================
st.title("📘 学習者レポート（6つのグラフ + AIインサイト）")

# --- Sidebar: 学習者選択 ---
student_ids = list(students.keys())
selected_id = st.sidebar.selectbox("学習者IDを選択してください", student_ids)

# --- Sidebar: インサイト言語 ---
language_label = st.sidebar.selectbox("インサイトの言語", ["日本語", "英語"])
lang_code = "ja" if language_label == "日本語" else "en"

# --- 選択された学習者の基本情報 ---
student = students[selected_id]

st.sidebar.markdown("### 👤 学習者情報")
st.sidebar.markdown(f"- **氏名：** {student['student_name']}")
st.sidebar.markdown(f"- **グループ：** {student['group_name']}")
st.sidebar.markdown(f"- **レベル：** {student['level']}")
st.sidebar.markdown(f"- **受講期間：** {student['period']}")

# --- ルールベース分析 ---
analysis = analyze_student(student)


# ============================================
# 🔹 グラフ 1行目：月末テスト & 出席・態度
# ============================================


# 左：グラフ1
col1, col2 = st.columns([1, 1])

with col1:
    graph3_learning_time(student)

with col2:
    # 中央揃え用の上下余白の追加
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    graph1_current_mastery(student)
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================
# 🔹 グラフ 2行目：学習時間 & テスト三角形
# ============================================

col3, col4 = st.columns(2)

with col3:
    graph2_attendance_radar(student)

with col4:
    graph5_test_triangle(student)


# ============================================
# 🔹 グラフ 3：課題・宿題
# ============================================

graph4_homework(student)


# ============================================
# 🔹 グラフ 4：学習進捗
# ============================================

graph6_progress(student)


# AI インサイト
st.markdown("---")
st.header("AIインサイト（自動生成）")

if st.button("AIインサイトを生成する"):
    with st.spinner("Llama 3.3 70B を呼び出してインサイトを生成しています..."):
        insights_text = generate_student_insights_llama(student, analysis, language=lang_code)
    st.text_area("インサイト結果", insights_text,height=400)
else:
    st.info("上のボタンをクリックすると、この学習者向けのAIインサイトが生成されます。")

