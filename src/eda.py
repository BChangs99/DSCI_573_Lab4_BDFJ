import altair as alt
import altair_ally as aly
alt.data_transformers.enable("vegafusion")

def save_distribution_charts(
        data, 
        col_list=None
    ):
    """
    Generate and save three Altair distribution charts as PNG:
      1. A bar chart showing the distributions of specified columns in 'data'.
      2. A bar chart showing the distribution of 'reviews_per_month' in 'data'.
      3. A correlation plot of features.

    Returns:
        None
    """

    if col_list is None:
        col_list = ['price', 'availability_365', 'number_of_reviews', 'minimum_nights']
    
    feature_dist_chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(alt.repeat('column'), bin=alt.Bin(maxbins=25)),
            y=alt.Y('count()')
        )
        .properties(width=140, height=200)
        .repeat(column=col_list)
    )

    target_dist_chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X('reviews_per_month', bin=alt.Bin(maxbins=50)),
            y=alt.Y('count()')
        )
        .properties(width=150, height=200)
    )

    feature_dist_chart.save("../results/figures/feature_dist_chart.png")
    target_dist_chart.save("../results/figures/target_dist_chart.png")
    aly.corr(data).save("../results/figures/corr.png")
