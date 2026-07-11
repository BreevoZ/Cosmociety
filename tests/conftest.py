import matplotlib

# Must happen before anything imports matplotlib.pyplot (cosmociety.visualize
# and cosmociety.animation both do at module level), otherwise matplotlib
# picks a GUI backend and tests fail/hang on machines without a display.
matplotlib.use("Agg")
