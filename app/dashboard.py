from src.recommenders.hybrid import (
    recommend_hybrid,
    DEFAULT_WEIGHTS,
)
from src.evaluation.feedback_analysis import analyze_feedback
from src.utils.feedback import save_recommendation_feedback
from pathlib import Path
import json
import joblib
import plotly.express as px
import pandas as pd
import streamlit as st



# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "raw"
REPORTS_DIR = ROOT_DIR / "reports"

MODELS_DIR = ROOT_DIR / "models"
ARTIFACT_PATH = MODELS_DIR / "recommender_artifacts.joblib"


# --------------------------------------------------
# 2. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="FinReco | Financial Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# 3. LOAD PROJECT DATA
# --------------------------------------------------

@st.cache_data
def load_csv(file_path):
    if not file_path.exists():
        return pd.DataFrame()

    return pd.read_csv(file_path)


@st.cache_data
def load_json(file_path):
    if not file_path.exists():
        return {}

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_resource
def load_recommender_artifacts():
    if not ARTIFACT_PATH.exists():
        return None

    return joblib.load(ARTIFACT_PATH)

customers = load_csv(DATA_DIR / "customers.csv")
products = load_csv(DATA_DIR / "products.csv")
interactions = load_csv(DATA_DIR / "interactions.csv")
api_metrics = load_csv(REPORTS_DIR / "api_monitoring.csv")

popularity = load_csv(REPORTS_DIR / "popularity_ranking.csv")
hybrid_recommendations = load_csv(
    REPORTS_DIR / "hybrid_recommendations.csv"
)

evaluation = load_json(REPORTS_DIR / "hybrid_evaluation.json")
quality_report = load_json(REPORTS_DIR / "data_quality_report.json")


# --------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------

st.sidebar.title("FinReco")
st.sidebar.caption("Financial Product Intelligence Platform")

page = st.sidebar.radio(
    "Navigation",
    [
    "Executive Overview",
    "Recommendation Analytics",
    "Personalized Explorer",
    "Customer Insights",
    "Product Analytics",
    "API Monitoring",
],
)

st.sidebar.divider()
st.sidebar.caption("Version 1.0 | Portfolio Project")


# --------------------------------------------------
# 5. DASHBOARD HEADER
# --------------------------------------------------

st.title("Financial Recommendation Intelligence")
st.caption(
    "A data-driven platform for personalized financial "
    "product recommendations and system monitoring."
)

st.divider()


# --------------------------------------------------
# 6. EXECUTIVE OVERVIEW
# --------------------------------------------------

if page == "Executive Overview":

    st.subheader("Executive Overview")

    total_customers = len(customers)
    total_products = len(products)
    available_products = int(products["available"].sum())
    total_interactions = len(interactions)

    consented_customers = (
        int(customers["consent"].sum())
        if "consent" in customers.columns
        else 0
    )

    total_requests = len(api_metrics)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Available Products", f"{available_products:,}")
    col3.metric("Interactions", f"{total_interactions:,}")
    col4.metric("Consented Customers", f"{consented_customers:,}")

    st.divider()


    st.subheader("Recommendation Model Performance")

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Precision @ 10",
        f"{evaluation.get('precision_at_k', 0):.3f}",
    )

    metric_columns[1].metric(
        "Recall @ 10",
        f"{evaluation.get('recall_at_k', 0):.3f}",
    )

    metric_columns[2].metric(
        "NDCG @ 10",
        f"{evaluation.get('ndcg_at_k', 0):.3f}",
    )

    metric_columns[3].metric(
        "Category Diversity",
        f"{evaluation.get('category_diversity', 0):.3f}",
    )

    st.caption(
        "Metrics represent offline evaluation on a synthetic "
        "temporal holdout dataset, not live business outcomes."
    )

    st.divider()

    st.subheader("Dataset Health")

    health_col1, health_col2, health_col3 = st.columns(3)

    health_col1.metric(
        "Customer Records",
        f"{len(customers):,}",
    )

    health_col2.metric(
        "Product Records",
        f"{len(products):,}",
    )

    health_col3.metric(
        "API Requests Recorded",
        f"{total_requests:,}",
    )

    st.success("Dashboard data sources loaded successfully.")


# --------------------------------------------------
# 7. OTHER PAGES — UNDER DEVELOPMENT
# --------------------------------------------------


elif page == "Recommendation Analytics":

    st.subheader("Recommendation Analytics")
    st.caption(
        "Offline recommendation analysis using the project's "
        "synthetic financial dataset."
    )

    # ----------------------------------------------
    # MODEL EVALUATION KPIs
    # ----------------------------------------------

    metric_cols = st.columns(4)

    metric_cols[0].metric(
        "Precision @ 10",
        f"{evaluation.get('precision_at_k', 0):.3f}",
    )

    metric_cols[1].metric(
        "Recall @ 10",
        f"{evaluation.get('recall_at_k', 0):.3f}",
    )

    metric_cols[2].metric(
        "NDCG @ 10",
        f"{evaluation.get('ndcg_at_k', 0):.3f}",
    )

    metric_cols[3].metric(
        "Catalog Coverage",
        f"{evaluation.get('catalog_coverage', 0):.1%}",
    )

    st.caption(
        "Evaluation uses a temporal holdout: 80% training "
        "and 20% testing across 100 evaluated users."
    )

    st.divider()

    # ----------------------------------------------
    # TOP POPULAR PRODUCTS
    # ----------------------------------------------

    st.subheader("Most Popular Financial Products")

    if not popularity.empty:

        top_products = popularity.nlargest(
            10, "interaction_count"
        ).sort_values("interaction_count", ascending=True)

        fig_popularity = px.bar(
            top_products,
            x="interaction_count",
            y="product_name",
            color="category",
            orientation="h",
            text="interaction_count",
            title="Top 10 Products by Interaction Count",
            labels={
                "interaction_count": "Interactions",
                "product_name": "Financial Product",
                "category": "Category",
            },
        )

        fig_popularity.update_layout(
            height=480,
            yaxis_title="",
            legend_title="Category",
        )

        st.plotly_chart(
            fig_popularity,
            use_container_width=True,
        )

    st.divider()

    # ----------------------------------------------
    # HYBRID RECOMMENDATION SCORES
    # ----------------------------------------------

    st.subheader("Hybrid Recommendation Score Analysis")

    if not hybrid_recommendations.empty:

        score_data = hybrid_recommendations.melt(
            id_vars=["product_name"],
            value_vars=[
                "popularity_score",
                "similarity_score",
                "collaborative_score",
                "hybrid_score",
            ],
            var_name="Scoring Method",
            value_name="Score",
        )

        fig_scores = px.bar(
            score_data,
            x="product_name",
            y="Score",
            color="Scoring Method",
            barmode="group",
            title="Recommendation Signal Comparison",
            labels={
                "product_name": "Financial Product",
                "Score": "Normalized Score",
            },
        )

        fig_scores.update_layout(
            height=500,
            xaxis_title="",
            yaxis_range=[0, 1.05],
            legend_title="Scoring Method",
        )

        st.plotly_chart(
            fig_scores,
            use_container_width=True,
        )

        st.caption(
            "These are the saved hybrid recommendation results, "
            "not a per-user or population-wide performance breakdown."
        )

    st.divider()

    # ----------------------------------------------
    # PRODUCT CATEGORY DISTRIBUTION
    # ----------------------------------------------

    st.subheader("Financial Product Category Distribution")

    if not products.empty:

        available_data = products[
            products["available"] == True
        ]

        category_data = (
            available_data["category"]
            .value_counts()
            .reset_index()
        )

        category_data.columns = ["Category", "Products"]

        fig_category = px.pie(
            category_data,
            names="Category",
            values="Products",
            hole=0.45,
            title="Available Products by Category",
        )

        fig_category.update_layout(height=450)

        st.plotly_chart(
            fig_category,
            use_container_width=True,
        )

elif page == "Personalized Explorer":

    st.subheader("Personalized Financial Product Explorer")

    st.caption(
        "Generate customer-specific financial product recommendations "
        "using popularity, content-based and collaborative filtering."
    )

    # ----------------------------------------------
    # CHECK REQUIRED DATA
    # ----------------------------------------------

    if customers.empty or products.empty or interactions.empty:

        st.error("Required customer, product or interaction data is missing.")

    else:

        artifacts = load_recommender_artifacts()

        if artifacts is None:

            st.error(
                "Recommendation artifacts not found. "
                "Run: python -m src.recommenders.build_artifacts"
            )

        else:

            # --------------------------------------
            # CUSTOMER SELECTION
            # --------------------------------------

            st.markdown("### 1. Select Customer")

            customer_options = sorted(
                customers["user_id"].astype(str).unique().tolist()
            )

            selected_user_id = st.selectbox(
                "Customer ID",
                customer_options,
                index=0,
                help="Select a customer to generate personalized recommendations."
            )

            customer_row = customers.loc[
                customers["user_id"].astype(str) == selected_user_id
            ].iloc[0]

            consent_granted = (
                str(customer_row["consent"]).strip().lower()
                in {"true", "1", "yes"}
            )

            profile_col1, profile_col2, profile_col3 = st.columns(3)

            profile_col1.metric(
                "Customer Segment",
                str(customer_row["customer_segment"]).replace("_", " ").title()
            )

            profile_col2.metric(
                "Age Group",
                str(customer_row["age_group"])
            )

            profile_col3.metric(
                "Recommendation Consent",
                "Granted" if consent_granted else "Not Granted"
            )

            st.caption(
                f"Customer interests: {str(customer_row['interests']).replace('|', ', ')}"
            )

            if consent_granted:
                st.success("Customer has granted recommendation consent.")
            else:
                st.warning(
                    "This customer has not granted recommendation consent. "
                    "The recommendation engine will not generate recommendations."
                )

            st.divider()

            # --------------------------------------
            # RECOMMENDATION SETTINGS
            # --------------------------------------

            st.markdown("### 2. Recommendation Settings")

            recommendation_limit = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=20,
                value=5,
            )

            generate_button = st.button(
                "Generate Personalized Recommendations",
                type="primary",
                use_container_width=True,
            )

            # --------------------------------------
            # GENERATE RECOMMENDATIONS
            # --------------------------------------

            if (
                generate_button
                or st.session_state.get(
                    "explorer_recommendations_user_id"
                ) == selected_user_id
            ):

                if not consent_granted:

                    st.error(
                        "Recommendations cannot be generated because "
                        "customer consent has not been granted."
                    )

                else:

                    if generate_button:
                        with st.spinner(
                            "Analyzing customer preferences and generating recommendations..."
                        ):
                            recommendations = recommend_hybrid(
                                user_id=selected_user_id,
                                customers=customers,
                                products=products,
                                interactions=interactions,
                                k=recommendation_limit,
                                artifacts=artifacts,
                            )

                        st.session_state["explorer_recommendations"] = (
                            recommendations
                        )

                        st.session_state["explorer_recommendations_user_id"] = (
                            selected_user_id
                        )
                    else:

                        recommendations = st.session_state[
                            "explorer_recommendations"
                        ]

                    st.divider()

                    st.markdown("### 3. Personalized Recommendations")

                    if recommendations.empty:

                        st.info(
                            "No eligible unseen financial products were found "
                            "for this customer."
                        )

                    else:

                        total_recommendations = len(recommendations)

                        avg_score = recommendations[
                            "hybrid_score"
                        ].mean()

                        result_col1, result_col2 = st.columns(2)

                        result_col1.metric(
                            "Recommendations Generated",
                            total_recommendations,
                        )

                        result_col2.metric(
                            "Average Hybrid Score",
                            f"{avg_score:.3f}",
                        )

                        st.caption(
                            "Higher hybrid scores indicate stronger ranking "
                            "under the configured recommendation model. "
                            "They are not probabilities of purchase."
                        )

                        # ----------------------------------
                        # RECOMMENDATION TABLE
                        # ----------------------------------

                        display_columns = [
                            "item_id",
                            "product_name",
                            "category",
                            "hybrid_score",
                            "recommendation_reason",
                        ]

                        display_data = recommendations[
                            display_columns
                        ].copy()

                        display_data["hybrid_score"] = (
                            display_data["hybrid_score"].round(4)
                        )

                        display_data["category"] = (
                            display_data["category"]
                            .str.replace("_", " ")
                            .str.title()
                        )

                        display_data.columns = [
                            "Product ID",
                            "Product Name",
                            "Category",
                            "Hybrid Score",
                            "Why Recommended",
                        ]

                        st.dataframe(
                            display_data,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ----------------------------------
                        # RECOMMENDATION SCORE CHART
                        # ----------------------------------

                        st.markdown("### 4. Recommendation Score Analysis")

                        chart_data = recommendations[
                            ["product_name", "hybrid_score"]
                        ].sort_values(
                            "hybrid_score",
                            ascending=True,
                        )

                        fig = px.bar(
                            chart_data,
                            x="hybrid_score",
                            y="product_name",
                            orientation="h",
                            text="hybrid_score",
                            title="Personalized Product Ranking",
                            labels={
                                "hybrid_score": "Hybrid Score",
                                "product_name": "Financial Product",
                            },
                        )

                        fig.update_traces(
                            texttemplate="%{text:.3f}",
                            textposition="outside",
                        )

                        fig.update_layout(
                            height=max(350, total_recommendations * 55),
                            showlegend=False,
                            xaxis_range=[
                                0,
                                max(1.1, float(chart_data["hybrid_score"].max()) * 1.15),
                            ],
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                        st.caption(
                            "Recommendations exclude previously interacted "
                            "products and products that fail availability "
                            "or age eligibility checks."
                        )


                        st.divider()

                        # ----------------------------------
                        # RECOMMENDATION EXPLAINABILITY
                        # ----------------------------------

                        st.markdown("### 5. Why Was This Product Recommended?")

                        product_ids = recommendations["item_id"].tolist()

                        product_names = recommendations.set_index(
                            "item_id"
                        )["product_name"].to_dict()

                        selected_product_id = st.selectbox(
                            "Select a recommended product",
                            product_ids,
                            format_func=lambda item_id: (
                                f"{item_id} - {product_names[item_id]}"
                            ),
                        )

                        selected_product = recommendations.loc[
                            recommendations["item_id"] == selected_product_id
                        ].iloc[0]

                        component_details = [
                            {
                                "Component": "Popularity",
                                "Score Column": "popularity_score",
                                "Weight": DEFAULT_WEIGHTS["popularity_score"],
                            },
                            {
                                "Component": "Interest Similarity",
                                "Score Column": "similarity_score",
                                "Weight": DEFAULT_WEIGHTS["similarity_score"],
                            },
                            {
                                "Component": "Collaborative Filtering",
                                "Score Column": "collaborative_score",
                                "Weight": DEFAULT_WEIGHTS["collaborative_score"],
                            },
                        ]

                        breakdown_rows = []

                        for component in component_details:

                            component_score = float(
                                selected_product[component["Score Column"]]
                            )

                            weight = component["Weight"]

                            contribution = component_score * weight

                            breakdown_rows.append({
                                "Component": component["Component"],
                                "Normalized Score": component_score,
                                "Model Weight": weight,
                                "Weighted Contribution": contribution,
                            })

                        breakdown = pd.DataFrame(breakdown_rows)

                        final_score = float(
                            selected_product["hybrid_score"]
                        )

                        st.markdown(
                            f"**Selected Product:** "
                            f"{selected_product['product_name']}"
                        )

                        st.metric(
                            "Final Hybrid Score",
                            f"{final_score:.4f}",
                        )

                        # ----------------------------------
                        # COMPONENT BREAKDOWN TABLE
                        # ----------------------------------

                        st.markdown("#### Score Contribution Breakdown")

                        display_breakdown = breakdown.copy()

                        for column in [
                            "Normalized Score",
                            "Model Weight",
                            "Weighted Contribution",
                        ]:
                            display_breakdown[column] = (
                                display_breakdown[column].round(4)
                            )

                        st.dataframe(
                            display_breakdown,
                            use_container_width=True,
                            hide_index=True,
                        )

                        # ----------------------------------
                        # CONTRIBUTION CHART
                        # ----------------------------------

                        fig_contribution = px.bar(
                            breakdown,
                            x="Component",
                            y="Weighted Contribution",
                            text="Weighted Contribution",
                            title="Contribution to Final Hybrid Score",
                        )

                        fig_contribution.update_traces(
                            texttemplate="%{text:.4f}",
                            textposition="outside",
                        )

                        fig_contribution.update_layout(
                            height=400,
                            yaxis_range=[
                                0,
                                max(
                                    0.45,
                                    float(
                                        breakdown[
                                            "Weighted Contribution"
                                        ].max()
                                    ) * 1.25,
                                ),
                            ],
                        )

                        st.plotly_chart(
                            fig_contribution,
                            use_container_width=True,
                        )

                        st.caption(
                            "Weighted contribution = normalized component "
                            "score × configured model weight. The sum of "
                            "these contributions equals the hybrid score."
                        )


                        st.divider()

                        # ----------------------------------
                        # CUSTOMER FEEDBACK
                        # ----------------------------------

                        st.markdown("### 6. Customer Feedback")

                        st.write(
                            "Help us understand whether this "
                            "recommendation is useful."
                        )

                        with st.form(
                            key="recommendation_feedback_form"
                        ):

                            feedback_choice = st.selectbox(
                                "Your feedback",
                                [
                                    "Select feedback",
                                    "Helpful",
                                    "Not Helpful",
                                    "Already Own This Product",
                                ],
                                key="recommendation_feedback_choice",
                            )

                            feedback_submit = st.form_submit_button(
                                "Submit Feedback"
                            )

                        if feedback_submit:

                            if feedback_choice == "Select feedback":

                                st.warning(
                                    "Please select your feedback first."
                                )

                            else:

                                save_recommendation_feedback(
                                    user_id=selected_user_id,
                                    item_id=selected_product_id,
                                    product_name=selected_product["product_name"],
                                    feedback=feedback_choice,
                                )

                                st.success(
                                    "Thank you! Your feedback has been saved."
                                )





elif page == "Customer Insights":

    st.subheader("Customer Insights")
    st.caption(
        "Understand customer segments, demographics, consent "
        "and financial product interaction behavior."
    )

    # ----------------------------------------------
    # CUSTOMER KPIs
    # ----------------------------------------------

    total_customers = len(customers)
    consented = int(customers["consent"].sum())
    non_consented = total_customers - consented

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Customers", f"{total_customers:,}")
    kpi2.metric("Consented", f"{consented:,}")
    kpi3.metric("Non-Consented", f"{non_consented:,}")
    kpi4.metric(
        "Consent Rate",
        f"{consented / total_customers:.1%}"
        if total_customers else "0%",
    )

    st.divider()

    # ----------------------------------------------
    # CUSTOMER SEGMENTS
    # ----------------------------------------------

    st.subheader("Customer Segment Distribution")

    segment_data = (
        customers["customer_segment"]
        .value_counts()
        .reset_index()
    )

    segment_data.columns = ["Customer Segment", "Customers"]

    fig_segments = px.bar(
        segment_data,
        x="Customer Segment",
        y="Customers",
        color="Customer Segment",
        text="Customers",
        title="Customer Distribution by Segment",
    )

    fig_segments.update_layout(
        height=420,
        showlegend=False,
        xaxis_title="",
    )

    st.plotly_chart(
        fig_segments,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------
    # AGE DISTRIBUTION
    # ----------------------------------------------

    st.subheader("Customer Age Distribution")

    age_data = (
        customers["age_group"]
        .value_counts()
        .reindex(
            ["18-24", "25-34", "35-44", "45-54", "55+"],
            fill_value=0,
        )
        .reset_index()
    )

    age_data.columns = ["Age Group", "Customers"]

    fig_age = px.bar(
        age_data,
        x="Age Group",
        y="Customers",
        color="Age Group",
        text="Customers",
        title="Customer Distribution by Age Group",
    )

    fig_age.update_layout(
        height=400,
        showlegend=False,
        xaxis_title="",
    )

    st.plotly_chart(
        fig_age,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------
    # CONSENT ANALYSIS
    # ----------------------------------------------

    st.subheader("Customer Consent Analysis")

    consent_data = pd.DataFrame({
        "Consent Status": ["Consented", "Not Consented"],
        "Customers": [consented, non_consented],
    })

    fig_consent = px.pie(
        consent_data,
        names="Consent Status",
        values="Customers",
        hole=0.5,
        title="Customer Consent Distribution",
    )

    fig_consent.update_layout(height=420)

    st.plotly_chart(
        fig_consent,
        use_container_width=True,
    )

    st.caption(
        "Consent status is shown as a customer-level data attribute. "
        "Recommendations must respect this preference."
    )

    st.divider()

    # ----------------------------------------------
    # INTERACTION BEHAVIOR
    # ----------------------------------------------

    st.subheader("Customer Interaction Behavior")

    event_data = (
        interactions["event_type"]
        .value_counts()
        .reindex(
            ["view", "click", "save", "enroll"],
            fill_value=0,
        )
        .reset_index()
    )

    event_data.columns = ["Event Type", "Interactions"]

    fig_events = px.bar(
        event_data,
        x="Event Type",
        y="Interactions",
        color="Event Type",
        text="Interactions",
        title="Customer Engagement by Event Type",
    )

    fig_events.update_layout(
        height=420,
        showlegend=False,
        xaxis_title="",
    )

    st.plotly_chart(
        fig_events,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------
    # CUSTOMER ACTIVITY OVER TIME
    # ----------------------------------------------

    st.subheader("Customer Activity Trend")

    activity_data = interactions.copy()

    activity_data["timestamp"] = pd.to_datetime(
        activity_data["timestamp"]
    )

    daily_activity = (
        activity_data
        .set_index("timestamp")
        .resample("D")
        .size()
        .reset_index(name="Interactions")
    )

    fig_activity = px.line(
        daily_activity,
        x="timestamp",
        y="Interactions",
        markers=True,
        title="Daily Customer Interactions",
        labels={
            "timestamp": "Date",
            "Interactions": "Total Interactions",
        },
    )

    fig_activity.update_layout(height=420)

    st.plotly_chart(
        fig_activity,
        use_container_width=True,
    )


elif page == "Product Analytics":

    st.subheader("Product Analytics")
    st.caption(
        "Analyze financial product availability, category mix, "
        "popularity and customer engagement."
    )

    # ----------------------------------------------
    # PRODUCT KPIs
    # ----------------------------------------------

    total_products = len(products)
    available_count = int(products["available"].sum())
    unavailable_count = total_products - available_count
    total_categories = products["category"].nunique()

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total Products", f"{total_products:,}")
    kpi2.metric("Available Products", f"{available_count:,}")
    kpi3.metric("Unavailable Products", f"{unavailable_count:,}")
    kpi4.metric("Product Categories", f"{total_categories:,}")

    st.divider()

    # ----------------------------------------------
    # CATEGORY FILTER
    # ----------------------------------------------

    st.subheader("Product Catalog Explorer")

    categories = ["All Categories"] + sorted(
        products["category"].dropna().unique().tolist()
    )

    selected_category = st.selectbox(
        "Filter by Product Category",
        categories,
    )

    if selected_category == "All Categories":
        filtered_products = products.copy()
    else:
        filtered_products = products[
            products["category"] == selected_category
        ].copy()

    st.caption(
        f"Showing {len(filtered_products)} products"
    )

    st.dataframe(
        filtered_products[
            [
                "item_id",
                "product_name",
                "category",
                "available",
                "minimum_age",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ----------------------------------------------
    # CATEGORY DISTRIBUTION
    # ----------------------------------------------

    st.subheader("Product Category Distribution")

    category_data = (
        products["category"]
        .value_counts()
        .reset_index()
    )

    category_data.columns = ["Category", "Products"]

    fig_categories = px.bar(
        category_data,
        x="Category",
        y="Products",
        color="Category",
        text="Products",
        title="Financial Products by Category",
    )

    fig_categories.update_layout(
        height=420,
        showlegend=False,
        xaxis_title="",
    )

    st.plotly_chart(
        fig_categories,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------
    # PRODUCT AVAILABILITY
    # ----------------------------------------------

    st.subheader("Product Availability Analysis")

    availability_data = pd.DataFrame({
        "Availability": ["Available", "Unavailable"],
        "Products": [available_count, unavailable_count],
    })

    fig_availability = px.pie(
        availability_data,
        names="Availability",
        values="Products",
        hole=0.5,
        title="Product Availability Distribution",
    )

    fig_availability.update_layout(height=420)

    st.plotly_chart(
        fig_availability,
        use_container_width=True,
    )

    st.divider()

    # ----------------------------------------------
    # TOP POPULAR PRODUCTS
    # ----------------------------------------------

    st.subheader("Top 10 Popular Financial Products")

    if not popularity.empty:

        filtered_popularity = popularity.copy()

        if selected_category != "All Categories":
            filtered_popularity = filtered_popularity[
                filtered_popularity["category"] == selected_category
            ]

        top_products = (
            filtered_popularity
            .nlargest(10, "interaction_count")
            .sort_values("interaction_count", ascending=True)
        )

        if not top_products.empty:

            fig_popular = px.bar(
                top_products,
                x="interaction_count",
                y="product_name",
                color="category",
                orientation="h",
                text="interaction_count",
                title="Products Ranked by Customer Interactions",
                labels={
                    "interaction_count": "Total Interactions",
                    "product_name": "Financial Product",
                    "category": "Category",
                },
            )

            fig_popular.update_layout(
                height=480,
                yaxis_title="",
                legend_title="Category",
            )

            st.plotly_chart(
                fig_popular,
                use_container_width=True,
            )

        else:
            st.info("No popularity data available for this category.")

    st.divider()

    # ----------------------------------------------
    # PRODUCT ENGAGEMENT TABLE
    # ----------------------------------------------

    st.subheader("Product Engagement Performance")

    if not popularity.empty:

        engagement_data = popularity.copy()

        if selected_category != "All Categories":
            engagement_data = engagement_data[
                engagement_data["category"] == selected_category
            ]

        engagement_data = engagement_data.sort_values(
            "interaction_count",
            ascending=False,
        )

        st.dataframe(
            engagement_data[
                [
                    "item_id",
                    "product_name",
                    "category",
                    "interaction_count",
                    "unique_users",
                    "popularity_score",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Interaction count measures recorded events, while "
            "unique users measures distinct customers interacting "
            "with each product. These are engagement indicators, "
            "not confirmed purchases."
        )


elif page == "API Monitoring":

    st.subheader("API Performance Monitoring")
    st.caption(
        "Monitor recommendation API latency, reliability, "
        "HTTP responses and recommendation caching."
    )

    # ----------------------------------------------
    # CHECK MONITORING DATA
    # ----------------------------------------------

    if api_metrics.empty:

        st.warning(
            "No API monitoring data found. "
            "Send requests to the recommendation API first."
        )

    else:

        monitoring = api_metrics.copy()

        monitoring["timestamp"] = pd.to_datetime(
            monitoring["timestamp"],
            utc=True,
        )

        monitoring["response_time_ms"] = pd.to_numeric(
            monitoring["response_time_ms"],
            errors="coerce",
        )

        monitoring["status_code"] = pd.to_numeric(
            monitoring["status_code"],
            errors="coerce",
        )

        monitoring["cache_hit"] = (
            monitoring["cache_hit"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
        )

        monitoring = monitoring.dropna(
            subset=["timestamp", "response_time_ms", "status_code"]
        )

        # ------------------------------------------
        # API PERFORMANCE KPIs
        # ------------------------------------------

        total_requests = len(monitoring)

        successful_requests = int(
            monitoring["status_code"].between(200, 299).sum()
        )

        failed_requests = total_requests - successful_requests

        avg_response_time = monitoring[
            "response_time_ms"
        ].mean()

        p95_response_time = monitoring[
            "response_time_ms"
        ].quantile(0.95)

        success_rate = (
            successful_requests / total_requests
            if total_requests else 0
        )

        cache_hits = int(
            monitoring["cache_hit"].eq(True).sum()
        )

        cache_misses = int(
            monitoring["cache_hit"].eq(False).sum()
        )

        cache_requests = cache_hits + cache_misses

        cache_hit_rate = (
            cache_hits / cache_requests
            if cache_requests else 0
        )

        # ------------------------------------------
        # KPI CARDS
        # ------------------------------------------

        st.subheader("API Performance Overview")

        kpi1, kpi2, kpi3 = st.columns(3)

        kpi1.metric(
            "Total Requests",
            f"{total_requests:,}",
        )

        kpi2.metric(
            "Average Response Time",
            f"{avg_response_time:.2f} ms",
        )

        kpi3.metric(
            "P95 Response Time",
            f"{p95_response_time:.2f} ms",
        )

        kpi4, kpi5, kpi6 = st.columns(3)

        kpi4.metric(
            "Success Rate",
            f"{success_rate:.1%}",
        )

        kpi5.metric(
            "Failed Requests",
            f"{failed_requests:,}",
        )

        kpi6.metric(
            "Cache Hit Rate",
            f"{cache_hit_rate:.1%}",
        )

        st.divider()

        # ------------------------------------------
        # RESPONSE TIME TREND
        # ------------------------------------------

        st.subheader("API Response Time Trend")

        response_data = monitoring.sort_values("timestamp")

        fig_response = px.line(
            response_data,
            x="timestamp",
            y="response_time_ms",
            markers=True,
            title="Response Time per API Request",
            labels={
                "timestamp": "Request Time (UTC)",
                "response_time_ms": "Response Time (ms)",
            },
        )

        fig_response.update_layout(height=420)

        st.plotly_chart(
            fig_response,
            use_container_width=True,
        )

        st.divider()

        # ------------------------------------------
        # HTTP STATUS DISTRIBUTION
        # ------------------------------------------

        st.subheader("HTTP Status Code Distribution")

        status_data = (
            monitoring["status_code"]
            .astype(int)
            .value_counts()
            .sort_index()
            .reset_index()
        )

        status_data.columns = ["Status Code", "Requests"]

        fig_status = px.bar(
            status_data,
            x="Status Code",
            y="Requests",
            color="Status Code",
            text="Requests",
            title="API Requests by HTTP Status Code",
        )

        fig_status.update_layout(
            height=400,
            showlegend=False,
        )

        st.plotly_chart(
            fig_status,
            use_container_width=True,
        )

        st.divider()

        # ------------------------------------------
        # CACHE PERFORMANCE
        # ------------------------------------------

        st.subheader("Recommendation Cache Performance")

        cache_col1, cache_col2 = st.columns(2)

        cache_data = pd.DataFrame({
            "Cache Status": ["Cache Hits", "Cache Misses"],
            "Requests": [cache_hits, cache_misses],
        })

        fig_cache = px.pie(
            cache_data,
            names="Cache Status",
            values="Requests",
            hole=0.5,
            title="Cache Hit vs Cache Miss",
        )

        fig_cache.update_layout(height=400)

        cache_col1.plotly_chart(
            fig_cache,
            use_container_width=True,
        )

        cache_comparison = (
            monitoring[monitoring["cache_hit"].isin([True, False])]
            .groupby("cache_hit")["response_time_ms"]
            .mean()
            .reset_index()
        )

        cache_comparison["Cache Status"] = (
            cache_comparison["cache_hit"]
            .map({True: "Cache Hit", False: "Cache Miss"})
        )

        fig_cache_time = px.bar(
            cache_comparison,
            x="Cache Status",
            y="response_time_ms",
            color="Cache Status",
            text_auto=".2f",
            title="Average Response Time by Cache Status",
            labels={
                "response_time_ms": "Average Response Time (ms)",
            },
        )

        fig_cache_time.update_layout(
            height=400,
            showlegend=False,
        )

        cache_col2.plotly_chart(
            fig_cache_time,
            use_container_width=True,
        )

        st.caption(
            "Cache performance is measured from recorded API requests. "
            "Latency comparisons are observational, not controlled benchmarks."
        )

        st.divider()

        # ------------------------------------------
        # RECENT API REQUESTS
        # ------------------------------------------

        st.subheader("Recent API Request Logs")

        recent_requests = monitoring.sort_values(
            "timestamp",
            ascending=False,
        ).head(20).copy()

        recent_requests["timestamp"] = (
            recent_requests["timestamp"]
            .dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        )

        st.dataframe(
            recent_requests[
                [
                    "timestamp",
                    "status_code",
                    "response_time_ms",
                    "cache_hit",
                    "recommendation_count",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These monitoring records include requests generated "
            "during API testing as well as manual API usage."
        )


elif page == "Feedback Analytics":

    st.title("Customer Feedback Analytics")
    st.caption(
        "Understand how customers respond to personalized financial recommendations."
    )

    feedback_metrics = analyze_feedback()

    total = feedback_metrics["total_feedback"]

    if total == 0:
        st.info(
            "No customer feedback available yet. "
            "Generate recommendations and submit feedback in Personalized Explorer."
        )

    else:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Feedback",
            f'{feedback_metrics["total_feedback"]:,}',
        )

        col2.metric(
            "Helpful Feedback",
            f'{feedback_metrics["helpful_count"]:,}',
        )

        col3.metric(
            "Not Helpful",
            f'{feedback_metrics["not_helpful_count"]:,}',
        )

        col4.metric(
            "Helpful Rate",
            f'{feedback_metrics["helpful_rate"]:.2f}%',
        )

        st.divider()

        st.subheader("Customer Feedback Distribution")

        feedback_distribution = pd.DataFrame({
            "Feedback": [
                "Helpful",
                "Not Helpful",
                "Already Own This Product",
            ],
            "Count": [
                feedback_metrics["helpful_count"],
                feedback_metrics["not_helpful_count"],
                feedback_metrics["already_owned_count"],
            ],
        })

        fig = px.pie(
            feedback_distribution,
            names="Feedback",
            values="Count",
            hole=0.45,
            title="Feedback Breakdown",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.subheader("Product-Level Feedback")

        product_feedback = feedback_metrics["product_feedback"]

        if not product_feedback.empty:

            feedback_columns = [
                col
                for col in [
                    "Helpful",
                    "Not Helpful",
                    "Already Own This Product",
                ]
                if col in product_feedback.columns
            ]

            if feedback_columns:
                product_chart_data = product_feedback.melt(
                    id_vars=["item_id", "product_name"],
                    value_vars=feedback_columns,
                    var_name="Feedback",
                    value_name="Count",
                )

                product_chart_data = product_chart_data[
                    product_chart_data["Count"] > 0
                ]

                fig = px.bar(
                    product_chart_data,
                    x="product_name",
                    y="Count",
                    color="Feedback",
                    barmode="group",
                    title="Customer Feedback by Product",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

            st.dataframe(
                product_feedback,
                use_container_width=True,
                hide_index=True,
            )


else:
    st.subheader(page)
    st.info("This analytics section will be implemented in the next steps.")


