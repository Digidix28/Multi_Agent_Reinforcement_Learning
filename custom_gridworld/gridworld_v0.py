from .env.gridworld_environment import GridworldEnv

def env(**kwargs):
    return GridworldEnv(**kwargs)
