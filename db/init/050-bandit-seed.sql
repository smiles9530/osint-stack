-- Bandit state initialization
\connect osint

-- Insert initial bandit state for prompt variants
INSERT INTO bandit_state(key, count, success, params)
VALUES
  ('prompt:variantA', 1, 1, '{"temperature":0.1, "model":"llama3.1:8b"}'),
  ('prompt:variantB', 1, 1, '{"temperature":0.2, "model":"llama3.1:8b"}'),
  ('prompt:variantC', 1, 1, '{"temperature":0.3, "model":"qwen2.5:7b"}')
ON CONFLICT (key) DO NOTHING;

-- Insert initial bandit state for source weights
INSERT INTO bandit_state(key, count, success, params)
VALUES
  ('source:reuters', 10, 8, '{"weight":1.0, "reliability":"high"}'),
  ('source:bbc', 10, 7, '{"weight":0.9, "reliability":"high"}'),
  ('source:aljazeera', 10, 6, '{"weight":0.8, "reliability":"medium"}'),
  ('source:ft', 10, 7, '{"weight":0.9, "reliability":"high"}'),
  ('source:ap', 10, 8, '{"weight":1.0, "reliability":"high"}')
ON CONFLICT (key) DO NOTHING;

-- Insert initial source metadata
INSERT INTO source_metadata(domain, reputation_score, paywall_detected, reliability_label, bias_score)
VALUES
  ('reuters.com', 0.9, FALSE, 'high', 0.5),
  ('bbc.co.uk', 0.9, FALSE, 'high', 0.4),
  ('aljazeera.com', 0.8, FALSE, 'medium', 0.6),
  ('ft.com', 0.9, TRUE, 'high', 0.5),
  ('apnews.com', 0.95, FALSE, 'high', 0.5)
ON CONFLICT (domain) DO NOTHING;
