import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------
df = pd.read_csv("policies.csv")
documents_df = pd.read_csv("documents_requirement.csv")
policy_buying_links_df = pd.read_csv("policy_buying_links.csv")

# -------------------------------------------------
# Encode Categorical Columns (UI Based)
# -------------------------------------------------
categorical_columns = [
    "job_role",
    "insurance_type",
    "time_period",
    "financial_goal",
    "existing_policy",
    "health_condition"
]
encoders = {}
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# -------------------------------------------------
# Feature Selection (Matches UI)
# -------------------------------------------------
X = df[
    [
        "job_role",
        "annual_salary",
        "insurance_type",
        "coverage_amount",
        "time_period",
        "financial_goal",
        "dependents",
        "existing_policy",
        "health_condition"
    ]
]

y = df["policy_id"]

# -------------------------------------------------
# Train ML Model
# -------------------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# --- Explanation Templates (10 distinct 2-line pairs) ---
explanation_templates = [
    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- This excellent '{p_name}' policy offers ₹{p_cov:,.0f} coverage, aligning well with your ₹{salary:,.0f} salary and '{f_goal}' goal (suited for '{p_f_goal}' objectives).\n"
        f"- It's also ideal given your '{job}' role (like '{p_j_role}' profiles), {dependents} dependents, '{p_e_policy}' existing policy status, and '{p_h_cond}' health for a '{p_t_period}' term.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Perfectly tailored, this '{p_name}' option provides ₹{p_cov:,.0f} coverage, complementing your ₹{salary:,.0f} salary and '{f_goal}' goal (often targeting '{p_f_goal}' needs).\n"
        f"- Its compatibility with your '{job}' (resembling '{p_j_role}' careers), {dependents} dependents, '{p_e_policy}' policy background, and '{p_h_cond}' status makes it a solid '{p_t_period}' period choice.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Consider this strong '{p_name}' policy with ₹{p_cov:,.0f} coverage; it matches your ₹{salary:,.0f} salary and '{f_goal}' aspirations (a good fit for '{p_f_goal}' objectives).\n"
        f"- With your '{job}' role (similar to '{p_j_role}' profiles), {dependents} dependents, '{p_e_policy}' status, and '{p_h_cond}' health, it's a fitting choice for a '{p_t_period}' duration.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- This highly-rated '{p_name}' policy provides robust ₹{p_cov:,.0f} coverage, a great match for your ₹{salary:,.0f} income and '{f_goal}' (ideal for '{p_f_goal}' needs).\n"
        f"- It suits your '{job}' occupation (similar to '{p_j_role}' professionals), supports {dependents} dependents, fits '{p_e_policy}' existing policy situation, and '{p_h_cond}' health over a '{p_t_period}' term.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- An excellent choice, this '{p_name}' policy offers ₹{p_cov:,.0f} protection, aligning with your ₹{salary:,.0f} salary and '{f_goal}' (excellent for '{p_f_goal}' planning).\n"
        f"- This plan is well-suited for your '{job}' profession (akin to '{p_j_role}' roles), {dependents} dependents, '{p_e_policy}' policy background, and '{p_h_cond}' health status for a '{p_t_period}' duration.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Opt for this reliable '{p_name}' policy with ₹{p_cov:,.0f} coverage, a good fit for your ₹{salary:,.0f} earnings and '{f_goal}' objective (perfect for '{p_f_goal}' strategies).\n"
        f"- Ideal for your '{job}' background (like '{p_j_role}' positions), {dependents} dependents, '{p_e_policy}' existing policy details, and '{p_h_cond}' health, spanning a '{p_t_period}' period.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- This top-tier '{p_name}' policy features ₹{p_cov:,.0f} coverage, a perfect complement to your ₹{salary:,.0f} salary and '{f_goal}' aims (designed for '{p_f_goal}' goals).\n"
        f"- Tailored for your '{job}' career (similar to '{p_j_role}' careers), {dependents} dependents, '{p_e_policy}' policy status, and '{p_h_cond}' health for a '{p_t_period}' term.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Discover this comprehensive '{p_name}' policy, offering ₹{p_cov:,.0f} coverage that harmonizes with your ₹{salary:,.0f} income and '{f_goal}' target (excellent for '{p_f_goal}' planning).\n"
        f"- Its benefits suit your '{job}' field (reminiscent of '{p_j_role}' profiles), {dependents} dependents, '{p_e_policy}' policy history, and '{p_h_cond}' status, applicable for a '{p_t_period}' duration.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Embrace this protective '{p_name}' policy with ₹{p_cov:,.0f} coverage; it aligns with your ₹{salary:,.0f} earnings and '{f_goal}' objectives (a solid choice for '{p_f_goal}' needs).\n"
        f"- Perfect for your '{job}' role (often found among '{p_j_role}' professionals), {dependents} dependents, '{p_e_policy}' policy situation, and '{p_h_cond}' health over a '{p_t_period}' period.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- A smart choice, this '{p_name}' policy delivers ₹{p_cov:,.0f} coverage, matching your ₹{salary:,.0f} salary and '{f_goal}' (well-suited for '{p_f_goal}' planning).\n"
        f"- It caters to your '{job}' background (related to '{p_j_role}' careers), {dependents} dependents, '{p_e_policy}' existing policy standing, and '{p_h_cond}' health for a '{p_t_period}' term.",

    # Additional 10 templates to make it 20
    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- This secure '{p_name}' plan provides ₹{p_cov:,.0f} protection, ideal for your ₹{salary:,.0f} income and '{f_goal}' (targeting '{p_f_goal}' security).\n"
        f"- It aligns with your '{job}' profession (like '{p_j_role}' occupations), {dependents} dependents, '{p_e_policy}' policy status, and '{p_h_cond}' health, offering coverage for a '{p_t_period}' duration.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Offering superior value, this '{p_name}' policy has ₹{p_cov:,.0f} coverage, perfectly suited for your ₹{salary:,.0f} salary and '{f_goal}' aspirations (a prime choice for '{p_f_goal}' goals).\n"
        f"- Its features support your '{job}' role (similar to '{p_j_role}' roles), {dependents} dependents, '{p_e_policy}' existing policy, and '{p_h_cond}' health profile over a '{p_t_period}' period.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Explore this outstanding '{p_name}' policy, delivering ₹{p_cov:,.0f} coverage that complements your ₹{salary:,.0f} income and '{f_goal}' objective (great for '{p_f_goal}' planning).\n"
        f"- It's an excellent match for your '{job}' background (such as '{p_j_role}' careers), {dependents} dependents, '{p_e_policy}' policy history, and '{p_h_cond}' status for a '{p_t_period}' term.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- This effective '{p_name}' policy provides ₹{p_cov:,.0f} coverage, tailored to your ₹{salary:,.0f} salary and '{f_goal}' (ideal for '{p_f_goal}' strategies).\n"
        f"- It's very suitable for your '{job}' occupation (common among '{p_j_role}' professionals), {dependents} dependents, '{p_e_policy}' existing policy situation, and '{p_h_cond}' health across a '{p_t_period}' duration.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Trust this robust '{p_name}' policy with ₹{p_cov:,.0f} protection, a solid choice for your ₹{salary:,.0f} earnings and '{f_goal}' aims (perfect for '{p_f_goal}' planning).\n"
        f"- Designed for your '{job}' field (similar to '{p_j_role}' roles), it caters to {dependents} dependents, '{p_e_policy}' policy background, and '{p_h_cond}' health, over a '{p_t_period}' period.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- An advantageous '{p_name}' policy, offering ₹{p_cov:,.0f} coverage that perfectly matches your ₹{salary:,.0f} income and '{f_goal}' objectives (excellent for '{p_f_goal}' strategies).\n"
        f"- Its compatibility with your '{job}' role (resembling '{p_j_role}' careers), {dependents} dependents, '{p_e_policy}' status, and '{p_h_cond}' health makes it an optimal '{p_t_period}' choice.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- This distinguished '{p_name}' policy features ₹{p_cov:,.0f} coverage, blending seamlessly with your ₹{salary:,.0f} salary and '{f_goal}' aims (crafted for '{p_f_goal}' ambitions).\n"
        f"- Ideal for your '{job}' career (like '{p_j_role}' positions), it supports {dependents} dependents, '{p_e_policy}' existing policy status, and '{p_h_cond}' health for a '{p_t_period}' term.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- A wise investment, this '{p_i_type}' policy provides ₹{p_cov:,.0f} coverage, tailored to your ₹{salary:,.0f} earnings and '{f_goal}' (an excellent fit for '{p_f_goal}' goals).\n"
        f"- It suits your '{job}' background (often among '{p_j_role}' professionals), {dependents} dependents, '{p_e_policy}' policy situation, and '{p_h_cond}' health, over a '{p_t_period}' period.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- Discover this personalized '{p_name}' policy with ₹{p_cov:,.0f} coverage, matching your ₹{salary:,.0f} income and '{f_goal}' target (perfect for '{p_f_goal}' planning).\n"
        f"- Its structure suits your '{job}' field (similar to '{p_j_role}' profiles), {dependents} dependents, '{p_e_policy}' policy history, and '{p_h_cond}' status, applicable for a '{p_t_period}' duration.",

    lambda p_name, p_id, p_i_type, p_cov, salary, f_goal, p_f_goal, job, p_j_role, dependents, p_e_policy, p_h_cond, p_t_period:
        f"Recommended policy **{p_name} ({p_id})**:\n"
        f"- This superior '{p_name}' policy delivers ₹{p_cov:,.0f} coverage, aligning with your ₹{salary:,.0f} salary and '{f_goal}' (well-suited for '{p_f_goal}' strategies).\n"
        f"- It thoughtfully caters to your '{job}' background (related to '{p_j_role}' careers), {dependents} dependents, '{p_e_policy}' existing policy standing, and '{p_h_cond}' health for a '{p_t_period}' term."
]

# -------------------------------------------------
# Function to display required documents and buying link
# -------------------------------------------------
def display_policy_details(policy_id, policy_name, documents_df, policy_buying_links_df):
    try:
        required_docs_str = documents_df[documents_df['policy_id'] == policy_id]['required_documents'].iloc[0]
        formatted_docs = required_docs_str.replace(';', ', ')
        print(f"Required documents for policy {policy_name}: {formatted_docs}")
    except IndexError:
        print(f"Required documents for policy {policy_name}: Not available")

    try:
        buying_link = policy_buying_links_df[policy_buying_links_df['policy_id'] == policy_id]['buying_link'].iloc[0]
        print(f"Buying Link for policy {policy_name}: {buying_link}\n")
    except IndexError:
        print(f"Buying Link for policy {policy_name}: Not available\n")

# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
def recommend_top_3_policies():
    # User input module 
    print("\n--- Enter User Details ---")
    name=input("Name: ")
    age = int(input("Age: "))
    job = input("Job: ")
    salary = float(input("Salary (₹): "))
    insurance_type = input("Insurance Type: ")
    coverage = float(input("Coverage Range (₹): "))
    time_period = input("Time Period: ")
    financial_goal = input("Financial Goal: ")
    dependents = int(input("Dependents: "))
    existing_policy = input("Existing Policy (Yes/No): ")
    health_condition = input("Health Condition: ")

    # Input processing module
    user_input = pd.DataFrame([[
        encoders["job_role"].transform([job])[0],
        salary,
        encoders["insurance_type"].transform([insurance_type])[0],
        coverage,
        encoders["time_period"].transform([time_period])[0],
        encoders["financial_goal"].transform([financial_goal])[0],
        dependents,
        encoders["existing_policy"].transform([existing_policy])[0],
        encoders["health_condition"].transform([health_condition])[0]
    ]], columns=X.columns)

    encoded_insurance_type = user_input['insurance_type'].iloc[0]

    probabilities = model.predict_proba(user_input)[0]
    policy_ids = model.classes_

    results = pd.DataFrame({
        "policy_id": policy_ids,
        "score": probabilities
    })

    # ml analysis Module
    eligible_policies_df = df[df['insurance_type'] == encoded_insurance_type]
    filtered_results = results[results['policy_id'].isin(eligible_policies_df['policy_id'])]
    
    # Ranking Module 
    top_3 = filtered_results.sort_values("score", ascending=False).head(3)
    recommended_policies_info = df[df["policy_id"].isin(top_3["policy_id"])]

    # Recommendation output Module
    print("\n✅ Top 3 Recommended Policies:")
    print(recommended_policies_info[["policy_id", "policy_name", "provider"]].to_string(index=False))

    # AI Explanation Module
    print("\n✅ Explanation for Recommendations")

    available_templates = list(explanation_templates)
    random.shuffle(available_templates)

    for i, (index, policy_row) in enumerate(recommended_policies_info.iterrows()):
        policy_name = policy_row['policy_name']
        policy_id = policy_row['policy_id']

        policy_job_role = encoders["job_role"].inverse_transform([policy_row['job_role']])[0]
        policy_insurance_type = encoders["insurance_type"].inverse_transform([policy_row['insurance_type']])[0]
        policy_time_period = encoders["time_period"].inverse_transform([policy_row['time_period']])[0]
        policy_financial_goal = encoders["financial_goal"].inverse_transform([policy_row['financial_goal']])[0]
        policy_existing_policy = encoders["existing_policy"].inverse_transform([policy_row['existing_policy']])[0]
        policy_health_condition = encoders["health_condition"].inverse_transform([policy_row['health_condition']])[0]

        if available_templates:
            selected_template_func = available_templates.pop(0)
            explanation = selected_template_func(policy_name, policy_id, policy_insurance_type, policy_row['coverage_amount'], salary, financial_goal, policy_financial_goal, job, policy_job_role, dependents, policy_existing_policy, policy_health_condition, policy_time_period)
        else:
            explanation = f"Recommended policy **{policy_name} ({policy_id})**:\n- No more unique explanation templates available. This policy is a general good fit based on your criteria."

        print(explanation)

    # Documents & Purchase Link Module 
    print("\n✅ Required Documents & Buying Links")
    for i, (index, policy_row) in enumerate(recommended_policies_info.iterrows()):
        policy_name = policy_row['policy_name']
        policy_id = policy_row['policy_id']
        display_policy_details(policy_id, policy_name, documents_df, policy_buying_links_df)


# -------------------------------------------------
# Run
# -------------------------------------------------
recommend_top_3_policies()
