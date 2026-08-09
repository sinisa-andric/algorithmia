package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	simulatedAnnealingTemperature    = 100.0
	simulatedAnnealingCoolingRate    = 0.99
	simulatedAnnealingMinTemperature = 1e-6
	simulatedAnnealingMaxSteps       = 10000
)

// SimulatedAnnealing minimizuje sphere funkciju koristeći simulirano kaljenje: prihvata pogoršanja sa verovatnoćom
// koja opada sa temperaturom, a temperatura se hladi po fiksnoj stopi svaki korak
// problem.Point je početna tačka pretrage
func SimulatedAnnealing(problem models.Problem) (result models.Result, err error) {

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

	temperature := simulatedAnnealingTemperature
	if v, ok := problem.Payload["temperature"].(float64); ok {
		temperature = v
	}

	coolingRate := simulatedAnnealingCoolingRate
	if v, ok := problem.Payload["cooling_rate"].(float64); ok {
		coolingRate = v
	}

	minTemperature := simulatedAnnealingMinTemperature
	if v, ok := problem.Payload["min_temperature"].(float64); ok {
		minTemperature = v
	}

	maxSteps := simulatedAnnealingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	energy := fn.Evaluate(point)

	steps := 0
	for ; steps < maxSteps && temperature >= minTemperature; steps++ {

		candidate := make([]float64, len(point))
		for i := range point {
			candidate[i] = point[i] + (rand.Float64()*2-1)*temperature
		}

		candidateEnergy := fn.Evaluate(candidate)
		delta := candidateEnergy - energy

		if delta < 0 || rand.Float64() < math.Exp(-delta/temperature) {
			point = candidate
			energy = candidateEnergy
		}

		temperature *= coolingRate

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, point, energy, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, point, energy, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "simulated_annealing",
		Point:      point,
		Value:      energy,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
