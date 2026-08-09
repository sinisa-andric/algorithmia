package models

// Problem opisuje problem optimizacije koji treba rešiti
type Problem struct {
	Method  string         `json:"method,omitempty"`
	Data    string         `json:"data,omitempty"`
	Point   []float64      `json:"point,omitempty"`
	Payload map[string]any `json:"payload,omitempty"`
}

// Result - rešenje Problem-a
type Result struct {
	Method     string            `json:"method,omitempty"`
	Point      []float64         `json:"point,omitempty"`
	Value      float64           `json:"value,omitempty"`
	Steps      int               `json:"steps,omitempty"`
	Function   string            `json:"function,omitempty"`
	Trajectory []TrajectoryPoint `json:"trajectory,omitempty"`
}

// TrajectoryPoint je jedna zabeležena tačka putanje pretrage — opciono se popunjava samo kad payload solvera
// sadrži "include_trajectory": true
type TrajectoryPoint struct {
	Step  int       `json:"step"`
	Point []float64 `json:"point"`
	Value float64   `json:"value"`
}
