package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	stochasticTunnelingTemperature    = 10.0
	stochasticTunnelingCoolingRate    = 0.995
	stochasticTunnelingGamma          = 0.001
	stochasticTunnelingMinTemperature = 1e-6
	stochasticTunnelingMaxSteps       = 10000
	stochasticTunnelingStepSize       = 0.1
)

// StochasticTunneling minimizuje sphere funkciju koristeći simulirano kaljenje sa tunelovanjem cilja: tunelovanje
// spljoštava barijere između minimuma i olakšava izlazak iz njih (f_tunnel = 1 - exp(-gamma*f(x)))
// problem.Point je početna tačka pretrage
func StochasticTunneling(problem models.Problem) (result models.Result, err error) {

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

	temperature := stochasticTunnelingTemperature
	if v, ok := problem.Payload["temperature"].(float64); ok {
		temperature = v
	}

	coolingRate := stochasticTunnelingCoolingRate
	if v, ok := problem.Payload["cooling_rate"].(float64); ok {
		coolingRate = v
	}

	gamma := stochasticTunnelingGamma
	if v, ok := problem.Payload["gamma"].(float64); ok {
		gamma = v
	}

	minTemperature := stochasticTunnelingMinTemperature
	if v, ok := problem.Payload["min_temperature"].(float64); ok {
		minTemperature = v
	}

	maxSteps := stochasticTunnelingMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	tunnelEnergy := 1 - math.Exp(-gamma*fn.Evaluate(point))

	best := append([]float64(nil), point...)
	bestEnergy := fn.Evaluate(point)

	steps := 0
	for ; steps < maxSteps && temperature >= minTemperature; steps++ {

		candidate := make([]float64, len(point))
		for i := range point {
			candidate[i] = point[i] + (rand.Float64()*2-1)*stochasticTunnelingStepSize
		}

		candidateEnergy := fn.Evaluate(candidate)
		candidateTunnelEnergy := 1 - math.Exp(-gamma*candidateEnergy)

		delta := candidateTunnelEnergy - tunnelEnergy

		if delta < 0 || rand.Float64() < math.Exp(-delta/temperature) {
			point = candidate
			tunnelEnergy = candidateTunnelEnergy

			if candidateEnergy < bestEnergy {
				bestEnergy = candidateEnergy
				best = append([]float64(nil), point...)
			}
		}

		temperature *= coolingRate

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestEnergy, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestEnergy, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "stochastic_tunneling",
		Point:      best,
		Value:      bestEnergy,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
