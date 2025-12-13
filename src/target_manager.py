

# ---- TARGET CONFIGURATION METHODS ----
# Only these methods need to know about the YOLO layer structure, so the rest of the project
# doesn't need layer specifics and everything else can be abstracted
#


class TargetManager:
    def __init__(self, interp):
        self.interp = interp
        self.get = None

    def set_specific(self, indices):
        module = self.interp.layers[indices[0]]
        hook_name = f"neurons_at_{indices[0]}"

        self.interp.register_hook(module, hook_name)

        def get_targets_helper():
            act = self.interp.activations[hook_name]
            if len(indices) == 1:
                return act[0]
            else:
                return act[0, indices[1:]]

        self.get = get_targets_helper

    def set_layers(self, layer_channel_indices):

        def prep():
            for layer_channel_idx in layer_channel_indices:
                layer_idx = layer_channel_idx[0]

                module = self.interp.layers[layer_idx]
                hook_name = f"conv{layer_idx}"

                self.interp.register_hook(module, hook_name)

        def get_helper():
            acts = []

            for layer_channel_idx in layer_channel_indices:
                layer_idx = layer_channel_idx[0]
                channels = layer_channel_idx[1]

                hook_name = f"conv{layer_idx}"
                act = self.interp.activations[hook_name]

                if channels is None:  # Case when we want all channels
                    curr = act[0]
                elif isinstance(channels, int):  # Case when we want only 1 specific channel
                    curr = act[0, channels]
                elif isinstance(channels, list):  # Case when we want multiple channels
                    curr = act[0, channels]
                else:
                    raise ValueError()

                acts.append(curr)

            return acts

        prep()
        self.get = get_helper

    def set_layer_batched(self, layer_idx):
        module = self.interp.layers[layer_idx]
        hook_name = f"layer{layer_idx}"

        self.interp.register_hook(module, hook_name)

        def helper():
            act = self.interp.activations[hook_name]
            return act

        self.get = helper

    def set_conv_layer(self, layer_idx, channels):

        """
        For Conv layers:
        self.activations["conv#"] has dimensions (B, C, H, W)
        batch should be 0 since there's only 1 image
        C is the channel

        Note that in Yolo11 these are the conv layers:
        0, 1, 3, 5, 7, 17, 20
        """

        module = self.interp.layers[layer_idx]
        hook_name = f"conv{layer_idx}"

        self.interp.register_hook(module, hook_name)

        def get_targets_conv_helper():
            act = self.interp.activations[hook_name]
            if channels is None:  # Case when we want all channels
                return act[0]
            elif isinstance(channels, int):  # Case when we want only 1 specific channel
                return act[0, channels]
            elif isinstance(channels, list):  # Case when we want multiple channels
                return act[0, channels]

        self.get = get_targets_conv_helper

    def set_conv0(self, channels):
        """
        Convenience method for conv layer 0
        """
        self.set_conv_layer(0, channels)

    def set_conv1(self, channels):
        """
        Convenience method for conv layer 1
        """
        self.set_conv_layer(1, channels)
