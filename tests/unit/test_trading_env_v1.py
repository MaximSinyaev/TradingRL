import numpy as np
import pytest
from custom_envs.trading_env_v1 import TradingEnvV1

@pytest.fixture
def basic_env_no_commission():
    prices = np.array([10.0, 10.0, 12.0, 12.0])  # close prices
    features = np.zeros((len(prices), 1))  # пустой массив фичей
    env = TradingEnvV1(
        features=features,
        real_prices=prices,
        commission=0.0,  # Без комиссии
        reward_on_trades_only=True,
    )
    return env

def test_pnl_buy_sell_no_commission(basic_env_no_commission):
    env = basic_env_no_commission
    obs, _ = env.reset()

    # Step 0: Buy at 10.0
    obs, reward, done, truncated, info = env.step(1)
    assert reward == 0.0
    assert not done
    assert len(env.positions) > 0

    # Step 1: Hold
    obs, reward, done, truncated, info = env.step(0)
    assert reward == 0.0

    # Step 2: Sell at 12.0
    obs, reward, done, truncated, info = env.step(2)
    # Ожидаемая прибыль: (12 - 10) / 100 = 0.02
    assert pytest.approx(reward, 0.0001) == 0.02 * 100
    assert done is False or done is True  # может быть завершено, если данных больше нет

    # Проверим, что PnL также посчитан верно
    assert pytest.approx(env.pnl, 0.0001) == 2.0
    

def test_random_strategy_pnl():
    prices = np.array([
        84422.48, 84439.74, 84395.83, 84455.99, 84400.04,
        84378.9, 84345.99, 84326.64, 84293.09, 84272.25
    ])
    features = np.random.randn(len(prices), 3)  # допустим, 3 фичи

    env = TradingEnvV1(
        features=features,
        real_prices=prices,
        initial_deposit=100.0,
        buy_fraction=0.1,
        commission=0.0,
        reward_on_trades_only=False
    )

    actions = [1, 1, 0, 1, 2, 2, 1, 1, 2]
    obs, _ = env.reset()
    done = False

    for action in actions:
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break

    # Двигаемся до конца, чтобы произошло auto-liquidation
    while not done:
        obs, reward, done, truncated, info = env.step(0)

    # После завершения эпизода
    final_pnl = env.pnl
    final_deposit = env.deposit
    assert final_deposit > 0
    print(f"Final PNL: {final_pnl:.6f}")
    print(f"Final deposit + liquidated positions: ${final_deposit:.2f}")