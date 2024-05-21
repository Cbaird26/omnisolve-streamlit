import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="OmniSolve 3.0: Universe Evolution", layout="wide")

# Universe Evolution
def simulate_universe_evolution(time_steps):
    if time_steps > 1000:
        st.error("Time steps too large! Please enter a value less than 1000.")
        return

    # Using a simple power law model for scale factor evolution in a matter-dominated universe
    t = np.linspace(1, time_steps, time_steps)
    scale_factor = t**(2/3)  # Scale factor proportional to t^(2/3) in a matter-dominated universe
    
    fig, ax = plt.subplots()
    ax.plot(t, scale_factor)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Scale Factor')
    ax.set_title('Evolution of the Universe (Matter-Dominated Era)')
    st.pyplot(fig)
    
    st.write("""
    ### Understanding the Evolution of the Universe Simulation
    #### Time Step Explanation
    A time step in a simulation is a discrete interval of time over which the state of the system is updated. In cosmological simulations, time steps represent incremental units of cosmic time, allowing us to model the evolution of the universe over a given period.
    
    #### The Model and the Slight Bend
    The model we used for the evolution of the universe in a matter-dominated era is based on the Friedmann equations. The scale factor \(a(t)\) describes how distances in the universe expand over time. For a matter-dominated universe, the scale factor \(a(t)\) follows a power-law relation:
    
    \[ a(t) \propto t^{2/3} \]
    
    This relation means that as cosmic time \(t\) increases, the scale factor grows as \(t^{2/3}\).
    
    #### Key Points About the Simulation:
    - **Initial Conditions**: At early times (small \(t\)), the scale factor grows slowly because the universe is young and compact.
    - **Later Times**: As time progresses, the scale factor increases more rapidly. However, because it's a \(t^{2/3}\) relationship, the growth rate isn't linear but rather follows a curve. This causes the slight bend or curve in the graph.
    
    #### The Slight Bend
    The slight bend you observe in the graph of the universe's evolution over 1000 time steps is a manifestation of the \(t^{2/3}\) relationship. Here’s what it means:
    - **Early Times**: Initially, the universe's expansion was slower.
    - **Later Times**: As time progresses, the universe expands more quickly.
    - **Non-linear Growth**: The curve represents non-linear growth typical of the matter-dominated era.
    
    ### What We Now Know:
    - **Cosmic Expansion**: The universe expands, and the rate of expansion depends on the dominant form of energy or matter. In the matter-dominated era, the expansion follows the \(t^{2/3}\) rule.
    - **Cosmological Models**: The simple power-law model helps us understand the large-scale structure of the universe. More detailed models may include other factors like dark energy, radiation, and more.
    - **Simulation Limitations**: While the simple model gives a good approximation, more sophisticated models would involve solving the full Friedmann equations considering various components (dark matter, dark energy, radiation).
    """)

# Main app function
def main():
    st.title("OmniSolve 3.0: The Universal Solution")
    st.sidebar.title("Navigation")
    options = ["Home", "Quantum Gravity Simulation", "Gravitational Wave Analysis", "Extra Dimensions Exploration", 
               "Modified Gravity Theories", "Dark Matter Simulation", "Black Hole Lensing", "Quantum Fields", 
               "Universe Evolution", "About"]
    choice = st.sidebar.radio("Go to", options)

    if choice == "Home":
        st.write("Welcome to OmniSolve 3.0, your universal solution for advanced simulations and analyses across various scientific domains.")
        st.write("Navigate through the options in the sidebar to explore different simulations and analyses.")
    
    elif choice == "Quantum Gravity Simulation":
        st.header("Quantum Gravity Simulation")
        st.write("This simulation models the dynamics of spin networks in a quantum gravity framework.")
        num_nodes = st.number_input("Enter number of nodes:", min_value=1, value=10, max_value=100)
        steps = st.number_input("Enter number of steps:", min_value=1, value=5, max_value=100)
        if st.button("Run Simulation"):
            G = create_spin_network(num_nodes)
            G = evolve_spin_network(G, steps)
            plot_network(G, "Quantum Gravity Spin Network")

    elif choice == "Gravitational Wave Analysis":
        st.header("Gravitational Wave Analysis")
        st.write("Analyze gravitational wave data using wavelet denoising and adaptive filtering.")
        uploaded_file = st.file_uploader("Upload a CSV file with gravitational wave data", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file).values.flatten()
            analyze_gravitational_wave(data)

    elif choice == "Extra Dimensions Exploration":
        st.header("Extra Dimensions Exploration")
        st.write("Explore the implications of extra dimensions in physics.")
        if st.button("Run Simulation"):
            geodesic_eq = simulate_extra_dimensions()
            st.write("Simulation complete. Geodesic equations displayed below:")
            for eq in geodesic_eq:
                st.latex(sp.latex(eq))

    elif choice == "Modified Gravity Theories":
        st.header("Modified Gravity Theories")
        st.write("Simulate and analyze modified gravity theories such as f(R) gravity.")
        if st.button("Run Simulation"):
            field_eq = simulate_modified_gravity()
            st.write("Simulation complete. Modified field equations displayed above.")

    elif choice == "Dark Matter Simulation":
        st.header("Dark Matter Simulation")
        st.write("Simulate the distribution of dark matter based on a given density profile.")
        radius_range = st.number_input("Enter radius range:", min_value=0.1, value=50.0, max_value=100.0)
        rho_0 = st.number_input("Enter central density (rho_0):", min_value=0.0, value=0.3)
        r_s = st.number_input("Enter scale radius (r_s):", min_value=0.1, value=10.0)
        if st.button("Run Simulation"):
            simulate_dark_matter_distribution(radius_range, rho_0, r_s)

    elif choice == "Black Hole Lensing":
        st.header("Black Hole Lensing")
        st.write("Visualize the gravitational lensing effect caused by a black hole.")
        mass = st.number_input("Enter mass of the black hole (kg):", min_value=1e20, value=1e30, max_value=1e32)
        distance = st.number_input("Enter distance of light source (m):", min_value=1e10, value=1e13, max_value=1e14)
        if st.button("Run Simulation"):
            black_hole_lensing(mass, distance)

    elif choice == "Quantum Fields":
        st.header("Quantum Fields")
        st.write("Simulate the behavior of a quantum field on a grid.")
        grid_size = st.number_input("Enter grid size:", min_value=10, value=50, max_value=100)
        time_steps = st.number_input("Enter number of time steps:", min_value=10, value=100, max_value=1000)
        if st.button("Run Simulation"):
            simulate_quantum_field(grid_size, time_steps)

    elif choice == "Universe Evolution":
        st.header("Universe Evolution")
        st.write("Simulate the evolution of the universe over time in a matter-dominated era.")
        time_steps = st.number_input("Enter number of time steps:", min_value=10, value=100, max_value=1000)
        if st.button("Run Simulation"):
            simulate_universe_evolution(time_steps)

    elif choice == "About":
        st.header("About OmniSolve 3.0")
        st.write("OmniSolve 3.0 is a comprehensive tool designed to integrate advanced simulations and analyses across various scientific domains.")
        st.write("Developed to facilitate research and understanding in fields such as quantum gravity, gravitational wave analysis, extra dimensions exploration, and more.")
        st.write("This tool leverages modern visualization techniques to present complex data and concepts in an accessible manner.")

if __name__ == "__main__":
    main()
