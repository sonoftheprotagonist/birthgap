import numpy as np
from scipy.integrate import quad

def calculate_m_value(
    global_ubi_amount, 
    national_ubi_amount,
    G=0.746,  # Gini coefficient
    m=0.5,    # Population split
    B_m=0.09, # Bottom m% wealth holders
    T_m=0.91, # Top m% wealth holders
    N=69000000,  # UK Population size
    W=12000000000000,  # UK total wealth (£)
    current_tax=277000000000  # Current tax revenue
):
    """
    Calculate m_value based on input parameters.
    
    Args:
        global_ubi_amount: Monthly global UBI amount in pounds
        national_ubi_amount: Monthly national UBI amount in pounds
        G: Gini coefficient (default: 0.746)
        m: Population split (default: 0.5)
        B_m: Bottom m% wealth holders (default: 0.09)
        T_m: Top m% wealth holders (default: 0.91)
        N: Population size (default: 69,000,000)
        W: Total wealth in pounds (default: £12 trillion)
        current_tax: Current tax revenue in pounds (default: £277 billion)
    """
    # Fixed parameters
    GLOBAL_POPULATION = 8000000000  # 8 billion people
    GLOBAL_WEALTH = 450000000000000000  # 450 trillion pounds
    UK_WEALTH = W  # Use input wealth

    # Calculate UK's global UBI contribution
    UK_WEALTH_SHARE = UK_WEALTH / GLOBAL_WEALTH
    UK_MONTHLY_CONTRIBUTION = global_ubi_amount * UK_WEALTH_SHARE
    ANNUAL_GLOBAL_UBI_COST = UK_MONTHLY_CONTRIBUTION * GLOBAL_POPULATION * 12

    # Calculate national UBI cost
    NET_MONTHLY_UBI = national_ubi_amount - UK_MONTHLY_CONTRIBUTION
    ANNUAL_NATIONAL_UBI_COST = NET_MONTHLY_UBI * N * 12

    # Calculate total revenue needed
    R_desired = current_tax + ANNUAL_GLOBAL_UBI_COST + ANNUAL_NATIONAL_UBI_COST

    # Calculate P and k parameters
    P = (1 + G) / (1 - G)
    n = 1 - m
    R_m = B_m / T_m
    a = m**P
    b = (1 - m)**(1 / P)
    c = n**P
    d = (1 - n)**(1 / P)
    k = (a - R_m + c * R_m) / (c * R_m - R_m + d * R_m + a + b - 1)

    # Define Lorenz curve function
    def lorenz_curve(x):
        normalized_x = x/N
        return (1 - k) * normalized_x**P + k * (1 - (1 - normalized_x)**(1 / P))

    # Initialize arrays
    x_values = [N]
    y_values = [lorenz_curve(N) * W]
    A_values = []
    m_desired_values = []

    # Calculate for all iterations
    for i in range(1, min(676, N + 1)):  # Upper bounded by N
        x_values.append(N - i)
        y_values.append(lorenz_curve(N - i) * W)
        A_k = y_values[i-1] - y_values[i]
        A_values.append(A_k)
        running_sum = sum(np.log(A_values[j] + 1) * A_values[j] - A_values[j] for j in range(i))
        m_desired = R_desired / running_sum
        m_desired_values.append(m_desired)

    # Print results
    print("\nInputs:")
    print(f"Global UBI: £{global_ubi_amount:.2f} per person per month")
    print(f"National UBI: £{national_ubi_amount:.2f} per person per month")
    print(f"Gini coefficient: {G}")
    print(f"Population split (m): {m}")
    print(f"Bottom {m*100}% wealth share: {B_m}")
    print(f"Top {m*100}% wealth share: {T_m}")
    print(f"Population size: {N:,}")
    print(f"Total wealth: £{W:,.2f}")
    
    print(f"\nResults:")
    print(f"P value: {P:.4f}")
    print(f"k value: {k:.4f}")
    print(f"Number of iterations: {len(m_desired_values)}")
    print(f"Final m_desired: {m_desired_values[-1]:.10f}")
    print(f"A_1 (wealth of richest person): £{A_values[0]:,.2f}")
    print(f"Total annual revenue needed: £{R_desired:,.2f}")

    return m_desired_values[-1]