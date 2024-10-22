import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "essentials.FluxAttentionSeeker",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("essentials")) {
            return;
        }

        if (nodeData.name === "FluxAttentionSeeker+") {
            const onCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                this.addWidget("button", "RESET ALL", null, () => {
                    this.widgets.forEach(w => {
                        if (w.type === "slider") {
                            w.value = 1.0;
                        }
                    });
                });

                this.addWidget("button", "ZERO ALL", null, () => {
                    this.widgets.forEach(w => {
                        if (w.type === "slider") {
                            w.value = 0.0;
                        }
                    });
                });

                this.addWidget("button", "REPEAT FIRST", null, () => {
                    var clip_value = undefined;
                    var t5_value = undefined;
                    this.widgets.forEach(w => {
                        if (w.name.startsWith('clip_l')) {
                            if (clip_value === undefined) {
                                clip_value = w.value;
                            }
                            w.value = clip_value;
                        } else if (w.name.startsWith('t5')) {
                            if (t5_value === undefined) {
                                t5_value = w.value;
                            }
                            w.value = t5_value;
                        }
                    });
                });
            };
        }
    },
});

app.registerExtension({
    name: "essentials.SD3AttentionSeekerLG",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("essentials")) {
            return;
        }

        if (nodeData.name === "SD3AttentionSeekerLG+") {
            const onCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                this.addWidget("button", "RESET L", null, () => {
                    this.widgets.forEach(w => {
                        if (w.type === "slider" && w.name.startsWith('clip_l')) {
                            w.value = 1.0;
                        }
                    });
                });
                this.addWidget("button", "RESET G", null, () => {
                    this.widgets.forEach(w => {
                        if (w.type === "slider" && w.name.startsWith('clip_g')) {
                            w.value = 1.0;
                        }
                    });
                });

                this.addWidget("button", "REPEAT FIRST", null, () => {
                    var clip_l_value = undefined;
                    var clip_g_value = undefined;
                    this.widgets.forEach(w => {
                        if (w.name.startsWith('clip_l')) {
                            if (clip_l_value === undefined) {
                                clip_l_value = w.value;
                            }
                            w.value = clip_l_value;
                        } else if (w.name.startsWith('clip_g')) {
                            if (clip_g_value === undefined) {
                                clip_g_value = w.value;
                            }
                            w.value = clip_g_value;
                        }
                    });
                });
            };
        }
    },
});

app.registerExtension({
    name: "essentials.SD3AttentionSeekerT5",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!nodeData?.category?.startsWith("essentials")) {
            return;
        }

        if (nodeData.name === "SD3AttentionSeekerT5+") {
            const onCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                this.addWidget("button", "RESET ALL", null, () => {
                    this.widgets.forEach(w => {
                        if (w.type === "slider") {
                            w.value = 1.0;
                        }
                    });
                });

                this.addWidget("button", "REPEAT FIRST", null, () => {
                    var t5_value = undefined;
                    this.widgets.forEach(w => {
                        if (w.name.startsWith('t5')) {
                            if (t5_value === undefined) {
                                t5_value = w.value;
                            }
                            w.value = t5_value;
                        }
                    });
                });
            };
        }
    },
});