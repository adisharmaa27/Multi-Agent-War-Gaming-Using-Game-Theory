from streamlit.testing.v1 import AppTest


def test_app_runs_without_errors():
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    assert len(at.tabs) == 5
    at.sidebar.slider[0].set_value(0.8).run()
    assert not at.exception
