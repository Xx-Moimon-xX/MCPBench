SCHEMA = """
create table competitors
(
    id                      int unsigned auto_increment comment 'unique identifier'
        primary key,
    competitor_name         varchar(50)   not null comment 'competitor name',
    car_series              varchar(50)   not null comment 'car series name',
    sales                   int           not null comment 'competitor sales',
    market_share_percentage decimal(5, 2) not null comment 'competitor market share percentage',
    record_date             date          not null comment 'record date'
)
    comment 'store competitor sales and market share' collate = utf8mb4_unicode_520_ci;

create table customer_flow
(
    id               int unsigned auto_increment comment 'unique identifier'
        primary key,
    region           varchar(50)   not null comment 'region',
    store            varchar(50)   not null comment 'store name',
    first_visit_flow int           not null comment 'first visit customer flow',
    total_visit_flow int           not null comment 'total customer flow',
    visit_datetime   datetime      not null comment 'visit time',
    conversion_rate  decimal(5, 2) not null comment 'conversion rate'
)
    comment 'store region, store, customer flow and conversion rate information' collate = utf8mb4_unicode_520_ci;

create index idx_region_store
    on customer_flow (region, store);

create table inventory
(
    id           int unsigned auto_increment comment 'unique identifier'
        primary key,
    car_series   varchar(50)  not null comment 'car series name',
    region       varchar(50)  not null comment 'region',
    warehouse    varchar(100) not null comment 'warehouse name',
    quantity     int          not null comment 'inventory quantity',
    last_checked datetime     not null comment 'last checked time',
    series_type  varchar(50)  not null comment 'car series type'
)
    comment 'store inventory information' collate = utf8mb4_unicode_520_ci;

create table market_sales
(
    id                      int unsigned auto_increment comment 'unique identifier'
        primary key,
    total_market_sales      int  not null comment 'total market sales',
    car_series_market_sales int  not null comment 'car series market sales',
    record_date             date not null comment 'record date'
)
    comment 'store market sales information' collate = utf8mb4_unicode_520_ci;

create table market_share
(
    id                      int unsigned auto_increment comment 'unique identifier'
        primary key,
    car_series              varchar(50)   not null comment 'car series name',
    market_share_percentage decimal(5, 2) not null comment 'market share percentage',
    record_date             date          not null comment 'record date'
)
    comment 'store car series market share changes' collate = utf8mb4_unicode_520_ci;

create table order_stats
(
    id                            int unsigned auto_increment comment 'unique identifier'
        primary key,
    car_series                    varchar(50) not null comment 'car series name',
    region                        varchar(50) not null comment 'region',
    order_quantity                int         not null comment 'order quantity',
    large_order_quantity          int         not null comment 'large order quantity',
    locked_order_quantity         int         not null comment 'locked order quantity',
    retained_large_order_quantity int         not null comment 'retained large order quantity'
)
    comment 'store order statistics data' collate = utf8mb4_unicode_520_ci;

create table policies
(
    id             int unsigned auto_increment comment 'unique identifier'
        primary key,
    policy_name    varchar(100) not null comment 'policy name',
    description    text         null comment 'policy description',
    type           varchar(50)  not null comment 'car series type',
    effective_date date         not null comment 'effective date',
    expiry_date    date         null comment 'expiry date'
)
    comment 'store national and local automotive industry policies' collate = utf8mb4_unicode_520_ci;

create table sales
(
    id          int unsigned auto_increment comment 'unique identifier'
        primary key,
    car_series  varchar(50) not null comment 'car series name',
    region      varchar(50) not null comment 'region',
    quantity    int         not null comment 'sales quantity',
    sale_date   date        not null comment 'sale date',
    series_type varchar(50) not null comment 'car series type'
)
    comment 'store actual sales data' collate = utf8mb4_unicode_520_ci;

create table sales_targets
(
    id             int unsigned auto_increment comment 'unique identifier'
        primary key,
    car_series     varchar(50) not null comment 'car series name',
    region         varchar(50) not null comment 'region',
    monthly_target int         not null comment 'monthly sales target',
    yearly_target  int         not null comment 'yearly sales target'
)
    comment 'store sales targets for each car series in each region' collate = utf8mb4_unicode_520_ci;
"""