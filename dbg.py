import dash, inspect, net_with_gif as appmodule
app = appmodule.app
print("Callbacks registered:", list(app.callback_map))
