from types import SimpleNamespace


def iq_learn_config():
    return SimpleNamespace(
        cuda_deterministic=False,
        device=None,
        gamma=0.99,
        seed=0,
        num_seed_steps=0,  # Don't need seeding for IL (Use 1000 for RL)
        only_expert_states=False,
        save_interval=5,  # Save networks every this many epochs
        eval_only=False,
        # Do offline learning
        offline=True,
        # Number of actor updates per env step
        num_actor_updates=1,
        pre_trained_encoder=True,
        # train
        train=SimpleNamespace(
            batch=16,
            use_target=True,
            soft_update=True,
        ),
        # method
        expert=SimpleNamespace(
            demos=2,
            subsample_freq=20,
        ),
        method=SimpleNamespace(
            type="iq",
            loss="value_expert",
            constrain=False,
            grad_pen=False,
            chi=True,
            tanh=False,
            regularize=False,
            div=None,
            alpha=0.5,
            lambda_gp=10,
            mix_coeff=1,
        ),
        # agent
        agent=SimpleNamespace(
            name="sac",
            init_temp=1e-2,  # use a low temp for IL
            alpha_lr=3e-4,
            alpha_betas=[0.9, 0.999],
            actor_lr=3e-4,
            actor_betas=[0.9, 0.999],
            actor_update_frequency=1,
            critic_lr=3e-4,
            critic_betas=[0.9, 0.999],
            critic_tau=0.005,
            critic_target_update_frequency=1,
            # learn temperature coefficient (disabled by default)
            learn_temp=False,
            # Use either value_dice actor or normal SAC actor loss
            vdice_actor=False,
        ),
        # q_net
        q_net=SimpleNamespace(_target_="agent.sac_models.DoubleQCritic"),
    )


def bc_config():
    return SimpleNamespace(
        cuda_deterministic=False,
        device=None,
        pre_trained_encoder=False,
        train=SimpleNamespace(
            batch=32,
        ),
    )
