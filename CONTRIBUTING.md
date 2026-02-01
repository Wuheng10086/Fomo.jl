# Contributing to ElasticWave2D.jl

Thank you for your interest in contributing to **ElasticWave2D.jl**! We welcome bug reports, feature requests, and pull requests.

## 🛠 Development Workflow

1.  **Fork** the repository.
2.  **Clone** your fork locally.
3.  **Create a branch** for your feature or fix (`git checkout -b feature/amazing-feature`).
4.  **Make changes** following the project structure and coding style.
5.  **Test** your changes using the tests in `tests/`.
6.  **Commit** your changes with clear messages (`git commit -m "feat: Add new boundary condition"`).
7.  **Push** to your fork (`git push origin feature/amazing-feature`).
8.  **Open a Pull Request**.

## 🎨 Coding Style

*   **Files**: Use `snake_case.jl` (e.g., `velocity_kernel.jl`).
*   **Modules/Types**: Use `PascalCase` (e.g., `VelocityModel`).
*   **Functions**: Use `snake_case` (e.g., `update_velocity!`).
*   **Directories**: Use `kebab-case` or `snake_case` matching the domain (e.g., `physics/wave_propagation`).
*   **Comments**: Add docstrings to all public functions and types.

## 🧪 Testing

Please ensure all tests pass before submitting a PR.
Run tests using:
```julia
julia --project=. -e 'using Pkg; Pkg.test()'
```
