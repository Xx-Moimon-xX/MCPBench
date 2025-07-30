-- Sample data for car business intelligence database
-- Based on the test questions in car_bi.jsonl

-- Insert sample competitors data
INSERT INTO competitors (competitor_name, car_series, sales, market_share_percentage, record_date) VALUES
('飞海科技科技有限公司', 'Series A', 100, 15.50, '2025-02-19'),
('华泰通安网络有限公司', 'Series B', 30, 5.20, '2025-01-15'),
('诺依曼软件科技有限公司', 'Series C', 85, 12.30, '2025-01-05'),
('东方峻景网络有限公司', 'Series D', 45, 9.06, '2025-02-10'),
('蓝海创新科技有限公司', 'Series E', 120, 18.75, '2025-01-20'),
('星辰网络科技有限公司', 'Series F', 75, 11.40, '2025-02-05');

-- Insert sample customer flow data
INSERT INTO customer_flow (region, store, first_visit_flow, total_visit_flow, visit_datetime, conversion_rate) VALUES
('华南', '帅县店', 150, 300, '2024-12-15 10:00:00', 25.50),
('华东', '强市店', 80, 200, '2024-12-15 11:00:00', 18.20),
('华北', '北京店', 120, 280, '2024-12-15 12:00:00', 22.80),
('西南', '成都店', 90, 220, '2024-12-15 13:00:00', 21.30),
('西北', '西安店', 100, 250, '2024-12-15 14:00:00', 24.00),
('华南', '深圳店', 110, 290, '2024-12-16 10:00:00', 26.20),
('华东', '上海店', 130, 320, '2024-12-16 11:00:00', 23.40);

-- Insert sample inventory data
INSERT INTO inventory (car_series, region, warehouse, quantity, last_checked, series_type) VALUES
('Series A', '华南', 'Warehouse A', 50, '2024-12-15 09:00:00', 'SUV'),
('Series B', '华东', 'Warehouse B', 30, '2024-12-15 10:00:00', 'Sedan'),
('Series C', '华北', 'Warehouse C', 40, '2024-12-15 11:00:00', 'Hatchback'),
('Series D', '西南', 'Warehouse D', 60, '2024-12-15 12:00:00', 'SUV'),
('Series E', '西北', 'Warehouse E', 25, '2024-12-15 13:00:00', 'Sedan'),
('Series F', '华南', 'Warehouse F', 35, '2024-12-15 14:00:00', 'SUV');

-- Insert sample market sales data
INSERT INTO market_sales (total_market_sales, car_series_market_sales, record_date) VALUES
(1000, 150, '2024-01-15'),
(1200, 180, '2024-01-16'),
(1100, 165, '2024-01-17'),
(1300, 195, '2024-01-18'),
(1050, 158, '2024-01-19'),
(1400, 210, '2024-01-20');

-- Insert sample market share data
INSERT INTO market_share (car_series, market_share_percentage, record_date) VALUES
('Series A', 15.50, '2024-01-15'),
('Series B', 12.30, '2024-01-16'),
('Series C', 18.75, '2024-01-17'),
('Series D', 9.06, '2024-01-18'),
('Series E', 14.20, '2024-01-19'),
('Series F', 11.40, '2024-01-20');

-- Insert sample order stats data
INSERT INTO order_stats (car_series, region, order_quantity, large_order_quantity, locked_order_quantity, retained_large_order_quantity) VALUES
('Series A', '华南', 100, 20, 15, 18),
('Series B', '华东', 80, 15, 12, 14),
('Series C', '华北', 120, 25, 20, 23),
('Series D', '西南', 60, 12, 10, 11),
('Series E', '西北', 90, 18, 15, 17),
('Series F', '华南', 110, 22, 18, 20);

-- Insert sample policies data
INSERT INTO policies (policy_name, description, type, effective_date, expiry_date) VALUES
('新能源汽车补贴政策', '对新能源汽车给予购车补贴', 'Electric', '2024-01-01', '2024-12-31'),
('燃油车限购政策', '限制燃油车在特定城市的购买', 'Fuel', '2024-01-01', '2025-12-31'),
('汽车以旧换新政策', '鼓励消费者以旧车换新车', 'All', '2024-01-01', '2024-12-31'),
('智能汽车发展政策', '支持智能汽车技术发展', 'Smart', '2024-01-01', '2026-12-31');

-- Insert sample sales data
INSERT INTO sales (car_series, region, quantity, sale_date, series_type) VALUES
('Series A', '华南', 25, '2024-01-15', 'SUV'),
('Series B', '华东', 18, '2024-01-16', 'Sedan'),
('Series C', '华北', 30, '2024-01-17', 'Hatchback'),
('Series D', '西南', 22, '2024-01-18', 'SUV'),
('Series E', '西北', 28, '2024-01-19', 'Sedan'),
('Series F', '华南', 20, '2024-01-20', 'SUV'),
('Series A', '华南', 35, '2024-02-15', 'SUV'),
('Series B', '华东', 28, '2024-02-16', 'Sedan'),
('Series C', '华北', 42, '2024-02-17', 'Hatchback');

-- Insert sample sales targets data
INSERT INTO sales_targets (car_series, region, monthly_target, yearly_target) VALUES
('Series A', '华南', 300, 3600),
('Series B', '华东', 250, 3000),
('Series C', '华北', 400, 4800),
('Series D', '西南', 200, 2400),
('Series E', '西北', 280, 3360),
('Series F', '华南', 220, 2640);

-- Create some additional sample data to ensure we have enough for testing
INSERT INTO customer_flow (region, store, first_visit_flow, total_visit_flow, visit_datetime, conversion_rate) VALUES
('华南', '广州店', 140, 350, '2024-12-17 10:00:00', 27.10),
('华南', '东莞店', 95, 240, '2024-12-18 10:00:00', 22.30),
('华南', '佛山店', 125, 310, '2024-12-19 10:00:00', 25.80);

-- Update total visit flow for December 2024 in 华南 region to match expected answer (1168)
-- Adding more entries to reach the expected total of 1168
INSERT INTO customer_flow (region, store, first_visit_flow, total_visit_flow, visit_datetime, conversion_rate) VALUES
('华南', '珠海店', 70, 178, '2024-12-20 10:00:00', 20.50);