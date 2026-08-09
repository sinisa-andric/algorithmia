package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	elephantPopulation = 20
	elephantMaxSteps   = 1000
	elephantTolerance  = 1e-6
	elephantRangeLow   = -5.0
	elephantRangeHigh  = 5.0
	elephantNumClans   = 5
	elephantAlpha      = 0.5
	elephantBeta       = 0.1
)

// Elephant minimizuje konfigurisanu benchmark funkciju koristeći Elephant Herding Optimization: populacija se deli u
// klanove, svaki slon se kreće ka matrijarhu (najboljem u klanu), matrijarh se dodatno pomera ka centru klana,
// a najgori slon u klanu se povremeno nasumično resetuje (separating)
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func Elephant(problem models.Problem) (result models.Result, err error) {

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

	population := elephantPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := elephantMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := elephantTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	numClans := elephantNumClans
	if v, ok := problem.Payload["num_clans"].(float64); ok {
		numClans = int(v)
	}
	numClans = max(1, min(numClans, population))

	alpha := elephantAlpha
	if v, ok := problem.Payload["alpha"].(float64); ok {
		alpha = v
	}

	beta := elephantBeta
	if v, ok := problem.Payload["beta"].(float64); ok {
		beta = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	elephants := make([][]float64, population)
	values := make([]float64, population)
	for i := range elephants {
		elephant := make([]float64, dimensions)
		for d := range elephant {
			elephant[d] = elephantRangeLow + rand.Float64()*(elephantRangeHigh-elephantRangeLow)
		}
		elephants[i] = elephant
		values[i] = fn.Evaluate(elephant)
	}

	clans := make([][]int, numClans)
	for i := range elephants {
		c := i % numClans
		clans[c] = append(clans[c], i)
	}

	best := append([]float64(nil), elephants[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), elephants[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for _, clan := range clans {
			if len(clan) == 0 {
				continue
			}

			matriarchIdx := clan[0]
			for _, idx := range clan {
				if values[idx] < values[matriarchIdx] {
					matriarchIdx = idx
				}
			}
			matriarchPos := append([]float64(nil), elephants[matriarchIdx]...)

			for _, i := range clan {
				for d := range elephants[i] {
					elephants[i][d] = elephants[i][d] + alpha*(matriarchPos[d]-elephants[i][d])*rand.Float64()
					elephants[i][d] = clamp(elephants[i][d], elephantRangeLow, elephantRangeHigh)
				}
				values[i] = fn.Evaluate(elephants[i])
			}

			center := make([]float64, dimensions)
			for _, idx := range clan {
				for d := range center {
					center[d] += elephants[idx][d]
				}
			}
			for d := range center {
				center[d] /= float64(len(clan))
			}
			for d := range elephants[matriarchIdx] {
				elephants[matriarchIdx][d] = clamp(beta*center[d], elephantRangeLow, elephantRangeHigh)
			}
			values[matriarchIdx] = fn.Evaluate(elephants[matriarchIdx])

			worstIdx := clan[0]
			for _, idx := range clan {
				if values[idx] > values[worstIdx] {
					worstIdx = idx
				}
			}
			for d := range elephants[worstIdx] {
				elephants[worstIdx][d] = elephantRangeLow + rand.Float64()*(elephantRangeHigh-elephantRangeLow)
			}
			values[worstIdx] = fn.Evaluate(elephants[worstIdx])
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), elephants[i]...)
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
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
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "elephant",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
