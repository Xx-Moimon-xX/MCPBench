-- PostgreSQL schema for car business intelligence database
-- Converted from MySQL schema in schema.py

CREATE TABLE competitors (
    id SERIAL PRIMARY KEY,
    competitor_name VARCHAR(50) NOT NULL,
    car_series VARCHAR(50) NOT NULL,
    sales INTEGER NOT NULL,
    market_share_percentage DECIMAL(5,2) NOT NULL,
    record_date DATE NOT NULL
);
COMMENT ON TABLE competitors IS 'store competitor sales and market share';
COMMENT ON COLUMN competitors.id IS 'unique identifier';
COMMENT ON COLUMN competitors.competitor_name IS 'competitor name';
COMMENT ON COLUMN competitors.car_series IS 'car series name';
COMMENT ON COLUMN competitors.sales IS 'competitor sales';
COMMENT ON COLUMN competitors.market_share_percentage IS 'competitor market share percentage';
COMMENT ON COLUMN competitors.record_date IS 'record date';

CREATE TABLE customer_flow (
    id SERIAL PRIMARY KEY,
    region VARCHAR(50) NOT NULL,
    store VARCHAR(50) NOT NULL,
    first_visit_flow INTEGER NOT NULL,
    total_visit_flow INTEGER NOT NULL,
    visit_datetime TIMESTAMP NOT NULL,
    conversion_rate DECIMAL(5,2) NOT NULL
);
COMMENT ON TABLE customer_flow IS 'store region, store, customer flow and conversion rate information';
COMMENT ON COLUMN customer_flow.id IS 'unique identifier';
COMMENT ON COLUMN customer_flow.region IS 'region';
COMMENT ON COLUMN customer_flow.store IS 'store name';
COMMENT ON COLUMN customer_flow.first_visit_flow IS 'first visit customer flow';
COMMENT ON COLUMN customer_flow.total_visit_flow IS 'total customer flow';
COMMENT ON COLUMN customer_flow.visit_datetime IS 'visit time';
COMMENT ON COLUMN customer_flow.conversion_rate IS 'conversion rate';

CREATE INDEX idx_region_store ON customer_flow (region, store);

CREATE TABLE inventory (
    id SERIAL PRIMARY KEY,
    car_series VARCHAR(50) NOT NULL,
    region VARCHAR(50) NOT NULL,
    warehouse VARCHAR(100) NOT NULL,
    quantity INTEGER NOT NULL,
    last_checked TIMESTAMP NOT NULL,
    series_type VARCHAR(50) NOT NULL
);
COMMENT ON TABLE inventory IS 'store inventory information';
COMMENT ON COLUMN inventory.id IS 'unique identifier';
COMMENT ON COLUMN inventory.car_series IS 'car series name';
COMMENT ON COLUMN inventory.region IS 'region';
COMMENT ON COLUMN inventory.warehouse IS 'warehouse name';
COMMENT ON COLUMN inventory.quantity IS 'inventory quantity';
COMMENT ON COLUMN inventory.last_checked IS 'last checked time';
COMMENT ON COLUMN inventory.series_type IS 'car series type';

CREATE TABLE market_sales (
    id SERIAL PRIMARY KEY,
    total_market_sales INTEGER NOT NULL,
    car_series_market_sales INTEGER NOT NULL,
    record_date DATE NOT NULL
);
COMMENT ON TABLE market_sales IS 'store market sales information';
COMMENT ON COLUMN market_sales.id IS 'unique identifier';
COMMENT ON COLUMN market_sales.total_market_sales IS 'total market sales';
COMMENT ON COLUMN market_sales.car_series_market_sales IS 'car series market sales';
COMMENT ON COLUMN market_sales.record_date IS 'record date';

CREATE TABLE market_share (
    id SERIAL PRIMARY KEY,
    car_series VARCHAR(50) NOT NULL,
    market_share_percentage DECIMAL(5,2) NOT NULL,
    record_date DATE NOT NULL
);
COMMENT ON TABLE market_share IS 'store car series market share changes';
COMMENT ON COLUMN market_share.id IS 'unique identifier';
COMMENT ON COLUMN market_share.car_series IS 'car series name';
COMMENT ON COLUMN market_share.market_share_percentage IS 'market share percentage';
COMMENT ON COLUMN market_share.record_date IS 'record date';

CREATE TABLE order_stats (
    id SERIAL PRIMARY KEY,
    car_series VARCHAR(50) NOT NULL,
    region VARCHAR(50) NOT NULL,
    order_quantity INTEGER NOT NULL,
    large_order_quantity INTEGER NOT NULL,
    locked_order_quantity INTEGER NOT NULL,
    retained_large_order_quantity INTEGER NOT NULL
);
COMMENT ON TABLE order_stats IS 'store order statistics data';
COMMENT ON COLUMN order_stats.id IS 'unique identifier';
COMMENT ON COLUMN order_stats.car_series IS 'car series name';
COMMENT ON COLUMN order_stats.region IS 'region';
COMMENT ON COLUMN order_stats.order_quantity IS 'order quantity';
COMMENT ON COLUMN order_stats.large_order_quantity IS 'large order quantity';
COMMENT ON COLUMN order_stats.locked_order_quantity IS 'locked order quantity';
COMMENT ON COLUMN order_stats.retained_large_order_quantity IS 'retained large order quantity';

CREATE TABLE policies (
    id SERIAL PRIMARY KEY,
    policy_name VARCHAR(100) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL,
    effective_date DATE NOT NULL,
    expiry_date DATE
);
COMMENT ON TABLE policies IS 'store national and local automotive industry policies';
COMMENT ON COLUMN policies.id IS 'unique identifier';
COMMENT ON COLUMN policies.policy_name IS 'policy name';
COMMENT ON COLUMN policies.description IS 'policy description';
COMMENT ON COLUMN policies.type IS 'car series type';
COMMENT ON COLUMN policies.effective_date IS 'effective date';
COMMENT ON COLUMN policies.expiry_date IS 'expiry date';

CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    car_series VARCHAR(50) NOT NULL,
    region VARCHAR(50) NOT NULL,
    quantity INTEGER NOT NULL,
    sale_date DATE NOT NULL,
    series_type VARCHAR(50) NOT NULL
);
COMMENT ON TABLE sales IS 'store actual sales data';
COMMENT ON COLUMN sales.id IS 'unique identifier';
COMMENT ON COLUMN sales.car_series IS 'car series name';
COMMENT ON COLUMN sales.region IS 'region';
COMMENT ON COLUMN sales.quantity IS 'sales quantity';
COMMENT ON COLUMN sales.sale_date IS 'sale date';
COMMENT ON COLUMN sales.series_type IS 'car series type';

CREATE TABLE sales_targets (
    id SERIAL PRIMARY KEY,
    car_series VARCHAR(50) NOT NULL,
    region VARCHAR(50) NOT NULL,
    monthly_target INTEGER NOT NULL,
    yearly_target INTEGER NOT NULL
);
COMMENT ON TABLE sales_targets IS 'store sales targets for each car series in each region';
COMMENT ON COLUMN sales_targets.id IS 'unique identifier';
COMMENT ON COLUMN sales_targets.car_series IS 'car series name';
COMMENT ON COLUMN sales_targets.region IS 'region';
COMMENT ON COLUMN sales_targets.monthly_target IS 'monthly sales target';
COMMENT ON COLUMN sales_targets.yearly_target IS 'yearly sales target';