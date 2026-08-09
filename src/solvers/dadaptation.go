package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	dadaptationLearningRate = 0.1
	dadaptationEpsilon      = 1e-8
	dadaptationMaxSteps     = 1000
	dadaptationTolerance    = 1e-6
	dadaptationStepClip     = 1.0
	// dadaptationDFloor=1e-6 je za ovaj budžet od 1000 koraka izazivao isto samopojačavajuće zamrzavanje kao
	// prodigy.go: x-x0 počinje na 0, pa je d = 0/gradSumNorm = 0 svaki put oboren na pod (floor) umesto da
	// realno raste (izmereno: x ostaje zaglavljen na inicijalnoj tački). Veći pod daje smislen prvi korak
	dadaptationDFloor = 1.0
)

// Dadaptation minimizuje konfigurisanu benchmark funkciju koristeći D-Adaptation: slično prodigy.go, ali sa
// jednostavnijom procenom skalara d (odnos udaljenosti od početne tačke i akumuliranog gradijenta) i BEZ ikakvog
// momentuma ili adaptivnog drugog momenta — čist SGD sa naučenim globalnim skalarom
// problem.Point je početna tačka pretrage
func Dadaptation(problem models.Problem) (result models.Result, err error) {

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

	learningRate := dadaptationLearningRate
	if v, ok := problem.Payload["learning_rate"].(float64); ok {
		learningRate = v
	}

	epsilon := dadaptationEpsilon
	if v, ok := problem.Payload["epsilon"].(float64); ok {
		epsilon = v
	}

	maxSteps := dadaptationMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := dadaptationTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	x0 := append([]float64(nil), problem.Point...)
	x := append([]float64(nil), x0...)
	gradSum := make([]float64, dimensions)

	best := append([]float64(nil), x...)
	bestValue := fn.Evaluate(x)

	maxNoImprove := max(maxSteps/2, 200)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		grad := numGrad(fn.Evaluate, x, proximalGradEps)
		for d := range gradSum {
			gradSum[d] += grad[d]
		}

		diff := make([]float64, dimensions)
		for d := range diff {
			diff[d] = x[d] - x0[d]
		}
		dEstimate := norm(diff) / (norm(gradSum) + epsilon)
		if dEstimate < dadaptationDFloor {
			dEstimate = dadaptationDFloor
		}

		delta := make([]float64, dimensions)
		for d := range delta {
			delta[d] = -dEstimate * learningRate * grad[d]
		}
		delta = clipStep(delta, dadaptationStepClip)
		for d := range x {
			x[d] += delta[d]
		}

		value := fn.Evaluate(x)
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), x...)
		}

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}
		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, fn.Evaluate(best), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "dadaptation",
		Point:      best,
		Value:      fn.Evaluate(best),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
