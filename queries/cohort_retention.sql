WITH cohorts AS (
    SELECT customer_id, substr(signup_date,1,7) signup_month FROM customers
), monthly AS (
    SELECT c.signup_month,
           CAST((julianday(s.event_date)-julianday(c.signup_month||'-01'))/30.44 AS INTEGER) months_since_signup,
           COUNT(DISTINCT CASE WHEN sub.status='active' OR date(sub.churn_date)>date(s.event_date) THEN c.customer_id END) active_customers,
           COUNT(DISTINCT c.customer_id) cohort_customers
    FROM cohorts c
    JOIN usage_events s ON s.customer_id=c.customer_id
    JOIN subscriptions sub ON sub.customer_id=c.customer_id
    GROUP BY c.signup_month,months_since_signup
)
SELECT signup_month,months_since_signup,
       active_customers*1.0/NULLIF(cohort_customers,0) retention_rate
FROM monthly
WHERE months_since_signup>=0
ORDER BY signup_month,months_since_signup;