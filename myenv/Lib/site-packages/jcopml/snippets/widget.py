import pyperclip
from textwrap import dedent
import ipywidgets as widgets
from IPython.display import display, clear_output

from .const import SNIPPETS, COPIED_HTML


class JCOpMLSnippetWidget:
    def __init__(self):
        # Components
        self.title = widgets.HTML(f"""
            <div style="line-height: 20px;">
                <span style="margin:0; padding:0; font-size: 18px; font-weight: bold">
                    {SNIPPETS["name"]}
                </span>    
                <br>
                <span style="margin:0; padding:0; font-size: 10px">
                    <em><sup>*untuk kebutuhan mengajar</sup></em>
                </span>
            </div>
        """)

        self.main_menu = widgets.Dropdown(options=SNIPPETS["menu"].keys(), layout=widgets.Layout(min_width="250px"))
        self.main_menu.observe(self.update_submenu, names='value')
        
        self.sub_menu = widgets.Dropdown(layout=widgets.Layout(min_width="250px", visibility="hidden"))

        self.copy_button = widgets.Button(description='Get Snippet', layout=widgets.Layout(width="100px"))
        self.copy_button.on_click(self.copy_snippet)

        self.notif = widgets.Output()

        # Widget Layout
        menu_box = widgets.VBox([self.main_menu, self.sub_menu])
        menu_box.layout.margin = '0px 10px'

        copy_box = widgets.VBox([self.copy_button, self.notif])

        # Display Widget
        display(widgets.HBox([self.title, menu_box, copy_box]))

    def copy_snippet(self, b):
        snippet = SNIPPETS["menu"][self.main_menu.value]
        if isinstance(snippet, dict):
            snippet = snippet[self.sub_menu.value]
            
        with self.notif:
            pyperclip.copy(dedent(snippet)[1:-1])
            clear_output(wait=True)
            display(COPIED_HTML)

    def update_submenu(self, change):
        snippet = SNIPPETS["menu"][change.new]
        if isinstance(snippet, dict):
            # Has sub-menu
            self.sub_menu.layout.visibility = "visible"
            self.sub_menu.options = snippet.keys()
            self.sub_menu.value = next(iter(snippet))
        else:
            self.sub_menu.layout.visibility = "hidden"
