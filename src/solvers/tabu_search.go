package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
)

const (
	tabuSearchStepSize = 0.5
	tabuSearchTabuSize = 10
	tabuSearchMaxSteps = 1000
)

// TabuSearch minimizuje sphere funkciju koristeći hill climbing sa tabu listom: čuva listu nedavno posećenih pozicija
// kako pretraga ne bi odmah ponovo posetila iste tačke
// problem.Point je početna tačka pretrage
func TabuSearch(problem models.Problem) (result models.Result, err error) {

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

	stepSize := tabuSearchStepSize
	if v, ok := problem.Payload["step_size"].(float64); ok {
		stepSize = v
	}

	tabuSize := tabuSearchTabuSize
	if v, ok := problem.Payload["tabu_size"].(float64); ok {
		tabuSize = int(v)
	}

	maxSteps := tabuSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	point := append([]float64(nil), problem.Point...)
	best := append([]float64(nil), point...)
	bestValue := fn.Evaluate(point)

	tabu := make([][]float64, 0, tabuSize)

	steps := 0
	for ; steps < maxSteps; steps++ {

		tabu = append(tabu, append([]float64(nil), point...))
		if len(tabu) > tabuSize {
			tabu = tabu[1:]
		}

		var candidateBest []float64
		candidateBestValue := math.Inf(1)

		for i := range point {
			for _, delta := range [2]float64{stepSize, -stepSize} {
				candidate := append([]float64(nil), point...)
				candidate[i] += delta

				if isTabu(candidate, tabu) {
					continue
				}

				candidateValue := fn.Evaluate(candidate)
				if candidateValue < candidateBestValue {
					candidateBestValue = candidateValue
					candidateBest = candidate
				}
			}
		}

		if candidateBest == nil {
			break
		}

		point = candidateBest
		if candidateBestValue < bestValue {
			bestValue = candidateBestValue
			best = append([]float64(nil), point...)
		}
		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "tabu_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func isTabu(candidate []float64, tabu [][]float64) bool {

	for _, visited := range tabu {
		equal := true
		for i := range candidate {
			if candidate[i] != visited[i] {
				equal = false
				break
			}
		}
		if equal {
			return true
		}
	}

	return false
}
