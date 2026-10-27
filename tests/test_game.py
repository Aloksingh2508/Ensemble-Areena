"""Game-logic tests + headless flows through every page and mission."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGE_FILES = sorted(str(p) for p in (ROOT / "app" / "pages").glob("*.py") if p.name != "__init__.py")


def page(name, **kw):
    return AppTest.from_file(str(ROOT / "app" / "pages" / name), default_timeout=kw.pop("timeout", 120))


def with_progress(at, **overrides):
    """Run the page once, then patch the player's progress."""
    at.run()
    game = dict(at.session_state["game"])
    game.update(overrides)
    at.session_state["game"] = game
    return at


def no_errors(at):
    assert not at.exception, [e.value for e in at.exception]


# ------------------------------------------------------------- smoke
def test_main_app_runs():
    no_errors(AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=60).run())


@pytest.mark.parametrize("path", PAGE_FILES)
def test_every_page_runs(path):
    no_errors(AppTest.from_file(path, default_timeout=120).run())


# --------------------------------------------------------- state logic
def _state_script():
    import streamlit as st
    from app.game import state

    state.init()
    state.complete_mission(1)
    state.complete_mission(1)  # second call must not pay twice
    st.session_state["after_first"] = dict(state.g())
    for ok in (True, True, True, False, True):
        state.record_answer(ok)


def test_xp_streak_and_achievements():
    at = AppTest.from_function(_state_script).run()
    no_errors(at)
    first = at.session_state["after_first"]
    assert first["completed"] == [1]
    assert first["xp"] == 100 + 100  # mission reward + "First Step" badge, paid once
    g = at.session_state["game"]
    assert g["best_streak"] == 3 and g["streak"] == 1
    assert "streak_starter" in g["achievements"] and "hot_streak" not in g["achievements"]


def _status_script():
    import streamlit as st
    from app.game import state

    state.init()
    st.session_state["s0"] = [state.status(i) for i in range(1, 7)]
    state.complete_mission(1)
    st.session_state["s1"] = [state.status(i) for i in range(1, 7)]
    st.session_state["lvl"] = state.level()


def test_missions_unlock_in_order():
    at = AppTest.from_function(_status_script).run()
    assert at.session_state["s0"] == ["in_progress"] + ["locked"] * 5
    assert at.session_state["s1"] == ["completed", "in_progress"] + ["locked"] * 4
    assert at.session_state["lvl"] == 2


def _import_script():
    import json as _json
    import streamlit as st
    from app.game import state

    state.init()
    good = _json.dumps({"player": "Zed<b>", "xp": 5, "completed": [1, 99], "achievements": ["first_step", "bogus"]})
    state.import_progress(good)
    st.session_state["loaded"] = dict(state.g())
    bad = []
    for payload in ('{"xp": "lots"}', "[1,2]", "not json", '{"completed": "x"}'):
        try:
            state.import_progress(payload)
            bad.append(payload)
        except ValueError:
            pass
    st.session_state["bad_accepted"] = bad
    state.set_name("  " + "x" * 50)
    st.session_state["name"] = state.g()["player"]
    st.session_state["safe"] = state.safe_name()


def test_import_validation_and_name_sanitising():
    at = AppTest.from_function(_import_script).run()
    no_errors(at)
    loaded = at.session_state["loaded"]
    assert loaded["completed"] == [1] and loaded["achievements"] == ["first_step"] and loaded["xp"] == 5
    assert at.session_state["bad_accepted"] == []
    assert len(at.session_state["name"]) == 20


def test_player_name_is_html_escaped():
    at = AppTest.from_function(lambda: __import__("app.game.state", fromlist=["x"]).set_name("<script>x</script>")).run()
    at2 = AppTest.from_function(_name_escape_script).run()
    assert "<script>" not in at2.session_state["safe"]


def _name_escape_script():
    import streamlit as st
    from app.game import state

    state.init()
    state.set_name("<script>x</script>")
    st.session_state["safe"] = state.safe_name()


# ------------------------------------------------------------ missions
def open_mission(n, completed=None, **extra):
    at = with_progress(page("missions.py", timeout=240), completed=completed or list(range(1, n)), **extra)
    at.session_state["active_mission"] = n
    return at.run()


def test_level1_flow_completes_mission():
    at = open_mission(1)
    at.slider(key="m1_depth").set_value(4).run()
    at.button(key="m1_train").click().run()
    no_errors(at)
    assert at.session_state["game"]["flags"].get("m1") is True
    at.radio(key="chk_1").set_value("Recall").run()
    at.button(key="chk_btn_1").click().run()
    no_errors(at)
    g = at.session_state["game"]
    assert g["completed"] == [1] and g["xp"] >= 100 and "first_step" in g["achievements"]


def test_level1_wrong_answer_resets_streak_and_does_not_complete():
    at = open_mission(1)
    at.slider(key="m1_depth").set_value(4).run()
    at.button(key="m1_train").click().run()
    at.radio(key="chk_1").set_value("Precision").run()
    at.button(key="chk_btn_1").click().run()
    assert at.session_state["game"]["completed"] == []


def test_level2_flow():
    at = open_mission(2)
    at.slider(key="m2_trees").set_value(40).run()
    at.button(key="m2_train").click().run()
    no_errors(at)
    at.radio(key="chk_2").set_value("Individual trees' random mistakes tend to cancel out").run()
    at.button(key="chk_btn_2").click().run()
    no_errors(at)
    assert 2 in at.session_state["game"]["completed"]


def test_level3_flow():
    at = open_mission(3)
    at.slider(key="m3_n").set_value(60).run()
    at.button(key="m3_train").click().run()
    no_errors(at)
    assert at.session_state["game"]["flags"].get("m3") is True, "AdaBoost should beat a single stump on the arena data"
    at.radio(key="chk_3").set_value("The samples earlier models got wrong").run()
    at.button(key="chk_btn_3").click().run()
    assert 3 in at.session_state["game"]["completed"]


def test_level4_voting_rounds():
    at = open_mission(4)
    no_errors(at)
    for _ in range(12):  # answer correctly by reading the revealed answer, then play on
        at.radio(key="m4_choice").set_value("A. Class 0").run()
        at.button(key="m4_submit").click().run()
        no_errors(at)
        s = at.session_state["m4"]
        if s["correct"] is False:
            pass
        if 4 in at.session_state["game"]["completed"]:
            break
        at.button(key="m4_next").click().run()
    g = at.session_state["game"]
    assert g["voting_correct"] >= 1  # at least some rounds are answerable by guessing Class 0
    assert g["xp"] >= 25


def test_level4_completes_after_three_correct():
    at = open_mission(4)
    for _ in range(40):
        cache, s = at.session_state["m4_cache"], at.session_state["m4"]
        P = cache["P"][:, s["idx"], :]
        import numpy as np
        ans = int(np.argmax(P.mean(axis=0))) if s["kind"] == "Soft" else int(np.argmax(P, axis=1).sum() >= 2)
        at.radio(key="m4_choice").set_value(f"{'AB'[ans]}. Class {ans}").run()
        at.button(key="m4_submit").click().run()
        no_errors(at)
        if 4 in at.session_state["game"]["completed"]:
            break
        at.button(key="m4_next").click().run()
    g = at.session_state["game"]
    assert 4 in g["completed"] and g["voting_correct"] == 3 and "voting_veteran" in g["achievements"]


def test_level5_flow():
    at = open_mission(5)
    at.button(key="m5_train").click().run()
    no_errors(at)
    at.radio(key="chk_5").set_value("So base models predict rows they have not seen (no leakage)").run()
    at.button(key="chk_btn_5").click().run()
    assert 5 in at.session_state["game"]["completed"]


def test_level6_wizard_and_grading():
    at = open_mission(6)
    for _ in range(3):  # dataset -> models -> configure -> train
        at.button(key="ch_next").click().run()
    no_errors(at)
    at.button(key="ch_train").click().run()
    no_errors(at)
    at.button(key="ch_next").click().run()  # compare
    at.button(key="ch_next").click().run()  # choose
    no_errors(at)
    best = max(at.session_state["champ"]["run"]["results"], key=lambda r: r.metrics["F1-score"]).name
    at.selectbox(key="ch_pick").set_value(best).run()
    at.button(key="ch_lock").click().run()
    no_errors(at)
    assert at.session_state["champ"]["result"]["ok"] is True
    assert 6 in at.session_state["game"]["completed"] and "champion" in at.session_state["game"]["achievements"]


def test_level6_wrong_pick_does_not_complete():
    at = open_mission(6)
    for _ in range(3):
        at.button(key="ch_next").click().run()
    at.button(key="ch_train").click().run()
    at.button(key="ch_next").click().run()
    at.button(key="ch_next").click().run()
    res = at.session_state["champ"]["run"]["results"]
    worst = min(res, key=lambda r: r.metrics["F1-score"])
    best = max(r.metrics["F1-score"] for r in res)
    if best - worst.metrics["F1-score"] <= 0.005:
        pytest.skip("models tied on this data")
    at.selectbox(key="ch_pick").set_value(worst.name).run()
    at.button(key="ch_lock").click().run()
    assert at.session_state["champ"]["result"]["ok"] is False
    assert 6 not in at.session_state["game"]["completed"]


def test_locked_mission_cannot_be_opened():
    at = page("missions.py", timeout=120)
    at.session_state["active_mission"] = 5
    at.run()
    assert at.session_state["active_mission"] == 1  # redirected to the current level


# --------------------------------------------------------------- quiz
def test_quiz_full_run_awards_xp_once():
    at = page("quiz.py", timeout=120).run()
    at.button(key="quiz_start").click().run()
    from app.game.content import QUIZ
    bank = {q[0]: q for q in QUIZ}
    for _ in range(5):
        qz = at.session_state["quiz"]
        _, _, options, answer, _ = bank[qz["ids"][qz["i"]]]
        at.radio(key="quiz_choice").set_value(options[answer]).run()
        at.button(key="quiz_submit").click().run()
        no_errors(at)
        at.button(key="quiz_next").click().run()
    no_errors(at)
    g = at.session_state["game"]
    assert at.session_state["quiz"]["done"] and at.session_state["quiz"]["score"] == 5
    assert g["quiz_best"] == 5 and {"quiz_master", "perfect_score", "hot_streak"} <= set(g["achievements"])
    assert len(g["solved"]) == 5


def test_quiz_swap_and_restart():
    at = page("quiz.py", timeout=120).run()
    at.button(key="quiz_start").click().run()
    initial_ids = list(at.session_state["quiz"]["ids"])
    initial_q0 = initial_ids[0]

    # Swap question
    at.button(key="quiz_swap").click().run()
    no_errors(at)
    new_q0 = at.session_state["quiz"]["ids"][0]
    assert new_q0 != initial_q0
    assert at.session_state["quiz"]["swaps"] == 1

    # Restart quiz
    at.button(key="quiz_restart").click().run()
    no_errors(at)
    assert at.session_state["quiz"]["i"] == 0
    assert at.session_state["quiz"]["score"] == 0
    assert at.session_state["quiz"]["swaps"] == 0



# ------------------------------------------------------- other pages
def test_theory_reading_unlocks_bookworm():
    at = page("theory.py").run()
    from app.game.content import THEORY
    for topic in list(THEORY)[:6]:
        at.radio(key="theory_topic").set_value(topic).run()
        no_errors(at)
    assert "theory_reader" in at.session_state["game"]["achievements"]
    at.radio(key="theory_topic").set_value("Interview Q&A").run()
    no_errors(at)


def test_playground_trains():
    at = page("playground.py", timeout=240).run()
    at.button(key="train_button").click().run()
    no_errors(at)
    assert at.session_state["last_run"]["results"]
    at.multiselect(key="pg_algos").set_value(["Decision Tree", "Voting Classifier", "Stacking Classifier", "Gradient Boosting"]).run()
    assert any("previous run" in w.value for w in at.warning)
    at.button(key="train_button").click().run()
    no_errors(at)
    assert at.session_state["game"]["trained"] == 2


def test_settings_reset():
    at = with_progress(page("settings.py"), xp=500, completed=[1, 2])
    at.checkbox(key="set_confirm").check().run()
    at.button(key="set_reset").click().run()
    no_errors(at)
    assert at.session_state["game"]["xp"] == 0 and at.session_state["game"]["completed"] == []
