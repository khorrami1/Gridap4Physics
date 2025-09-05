using GLMakie
using Observables
using Colors   # RGBA constructor
using Distributions  # Exponential distribution

# # -- License Verification Logic --
# using HTTP
# using JSON

# """
# Verify the license key by contacting the licensing server.
# Returns true if the key is valid, false otherwise.
# """
# function verify_license(key::String)
#     try
#         resp = HTTP.post("https://your-license-server.com/verify";
#                          headers = Dict("Content-Type" => "application/json"),
#                          body = JSON.json(Dict("key" => key)))
#         result = JSON.parse(String(resp.body))
#         return get(result, "valid", false)
#     catch err
#         println("License server unreachable: ", err)
#         return false
#     end
# end

# # Prompt for license key at startup
# println("Enter your license key:")
# license_key = chomp(readline())
# if !verify_license(license_key)
#     println("Invalid or expired license key. Exiting application.")
#     exit(1)
# end

# --------------------------------


function create_complex_gui()
    # Observables for parameters
    dist_type = Observable("normal")
    num_samples = Observable(500)
    bin_count = Observable(30)
    # Color channel observables
    r = Observable(0.2)
    g = Observable(0.4)
    b = Observable(0.6)
    a = Observable(1.0)
    # Combine channels into RGBA color
    color = lift(r, g, b, a) do rr, gg, bb, aa
        RGBA(rr, gg, bb, aa)
    end

    # Data generation based on distribution and sample count
    data = lift(dist_type, num_samples) do d, n
        if d == "normal"
            randn(n)
        elseif d == "uniform"
            rand(n)
        else  # exponential
            rand(Exponential(1.0), n)
        end
    end

    # Create figure and layout with spacing
    fig = Figure(size = (1000, 600))
    grid = GridLayout(spacing = (10, 10))
    fig[1, 1] = grid

    # Distribution selection menu
    grid[1, 1] = Label(fig.scene, "Distribution:")
    grid[1, 2] = Menu(fig.scene; options = ["normal", "uniform", "exponential"], selection = dist_type)

    # Samples slider
    grid[2, 1] = Label(fig.scene, "Samples:")
    slider_samples = Slider(fig.scene; range = 100:100:5000, startvalue = num_samples[])
    on(slider_samples.value) do val
        num_samples[] = Int(round(val))
    end
    grid[2, 2] = slider_samples

    # Bins slider
    grid[3, 1] = Label(fig.scene, "Bins:")
    slider_bins = Slider(fig.scene; range = 5:5:100, startvalue = bin_count[])
    on(slider_bins.value) do val
        bin_count[] = Int(round(val))
    end
    grid[3, 2] = slider_bins

    # RGBA channel sliders
    grid[4, 1] = Label(fig.scene, "Red:")
    slider_r = Slider(fig.scene; range = 0.0:0.01:1.0, startvalue = r[])
    on(slider_r.value) do val
        r[] = round(val; digits = 2)
    end
    grid[4, 2] = slider_r

    grid[5, 1] = Label(fig.scene, "Green:")
    slider_g = Slider(fig.scene; range = 0.0:0.01:1.0, startvalue = g[])
    on(slider_g.value) do val
        g[] = round(val; digits = 2)
    end
    grid[5, 2] = slider_g

    grid[6, 1] = Label(fig.scene, "Blue:")
    slider_b = Slider(fig.scene; range = 0.0:0.01:1.0, startvalue = b[])
    on(slider_b.value) do val
        b[] = round(val; digits = 2)
    end
    grid[6, 2] = slider_b

    grid[7, 1] = Label(fig.scene, "Alpha:")
    slider_a = Slider(fig.scene; range = 0.0:0.01:1.0, startvalue = a[])
    on(slider_a.value) do val
        a[] = round(val; digits = 2)
    end
    grid[7, 2] = slider_a

    # Axes for plots
    ax_hist = grid[1:7, 3] = Axis(fig.scene; title = "Histogram")
    ax_scatter = grid[8, 1:3] = Axis(fig.scene; title = "Scatter Plot")

    # Scatter data observable
    scatter_data = lift(num_samples) do n
        (randn(n), randn(n))
    end

    # Update histogram when data, bins, or color change
    lift(data, bin_count, color) do d, bins, col
        for plt in copy(ax_hist.scene.plots)
            delete!(ax_hist.scene, plt)
        end
        hist!(ax_hist, d; bins = bins, color = col)
    end

    # Update scatter when sample count or color change
    lift(scatter_data, color) do pts, col
        for plt in copy(ax_scatter.scene.plots)
            delete!(ax_scatter.scene, plt)
        end
        scatter!(ax_scatter, pts[1], pts[2]; color = col, markersize = 8)
    end

    return fig
end

# Launch GUI
fig = create_complex_gui()
# Display the GUI and block until user input to keep window open
display(fig)
println("Press Enter to exit GUI...")
readline()
