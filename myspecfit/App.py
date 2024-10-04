from Tools import init_session_state
from st_pages import Page, add_page_title, show_pages


show_pages(
    [
        Page("pages/Home.py", "Home", "🏠"),
        Page("pages/Spectrum.py", "Spectrum", "🔭"),
        Page("pages/Model.py", "Model", "🌈"),
        Page("pages/Fit.py", "Fit", "⚖️"),
        Page("pages/Analyse.py", "Analyse", "📝"),
        Page("pages/Plot.py", "Plot", "🎨"),
        Page("pages/Calculate.py", "Calculate", "🔢"),
    ]
)

add_page_title()

init_session_state()
