package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
)

const (
	evolutionStrategySigma       = 1.0
	evolutionStrategyMaxSteps    = 1000
	evolutionStrategyTolerance   = 1e-6
	evolutionStrategyWindow      = 10
	evolutionStrategyTargetRatio = 0.2
)

// EvolutionStrategy minimizuje sphere funkciju koristeći (1+1) strategiju evolucije: jedno potomče se generiše
// Gausovom mutacijom oko roditelja, a veličina koraka sigma se prilagođava Rechenberg-ovim pravilom uspeha 1/5
// problem.Point je početna tačka pretrage
func EvolutionStrategy(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	sigma := evolutionStrategySigma
	if v, ok := problem.Payload["sigma"].(float64); ok {
		sigma = v
	}

	maxSteps := evolutionStrategyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := evolutionStrategyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	dimensions := len(point)
	successCount := 0

	steps := 0
	for ; steps < maxSteps && fn.Evaluate(point) >= tolerance; steps++ {

		offspring := make([]float64, dimensions)
		for d := range offspring {
			offspring[d] = point[d] + sigma*randNorm()
		}

		if fn.Evaluate(offspring) < fn.Evaluate(point) {
			point = offspring
			successCount++
		}

		if (steps+1)%evolutionStrategyWindow == 0 {
			ratio := float64(successCount) / float64(evolutionStrategyWindow)
			if ratio > evolutionStrategyTargetRatio {
				sigma *= 1.2
			}
			if ratio < evolutionStrategyTargetRatio {
				sigma *= 0.8
			}
			successCount = 0
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, fn.Evaluate(point), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "evolution_strategy",
		Point:      point,
		Value:      fn.Evaluate(point),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
