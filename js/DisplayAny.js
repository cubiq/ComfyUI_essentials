import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "essentials.DisplayAny",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("essentials")) {
            return;
        }

        if (nodeData.name === "DisplayAny") {
            const onExecuted = nodeType.prototype.onExecuted;

            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
					for (let i = 1; i < this.widgets.length; i++) {
						this.widgets[i].onRemove?.();
					}
					this.widgets.length = 1;
				}

                // Check if the "text" widget already exists.
                let textWidget = this.widgets && this.widgets.find(w => w.name === "displaytext");
                if (!textWidget) {
                    textWidget = ComfyWidgets["STRING"](this, "displaytext", ["STRING", { multiline: true }], app).widget;
                    textWidget.inputEl.readOnly = true;
                    textWidget.inputEl.style.border = "none";
                    textWidget.inputEl.style.backgroundColor = "transparent";
                }
                textWidget.value = message["text"].join("");
            };
        }
    },
});