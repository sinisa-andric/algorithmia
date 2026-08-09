package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	harmonySearchMemorySize = 20
	harmonySearchHMCR       = 0.9
	harmonySearchPAR        = 0.3
	harmonySearchBandwidth  = 0.5
	harmonySearchMaxSteps   = 1000
	harmonySearchRange      = 15.0
	harmonySearchTolerance  = 1e-6
)

// HarmonySearch minimizuje sphere funkciju koristeći harmony search: kandidatska rešenja se improvizuju iz
// harmonijske memorije (sa verovatnoćom hmcr, povremeno fino podešena po visini) ili generišu nasumično,
// i zamenjuju najgoreg člana memorije kad god ga poboljšaju
// problem.Point samo određuje dimenzionalnost
func HarmonySearch(problem models.Problem) (result models.Result, err error) {

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

	memorySize := harmonySearchMemorySize
	if v, ok := problem.Payload["harmony_memory_size"].(float64); ok {
		memorySize = int(v)
	}

	hmcr := harmonySearchHMCR
	if v, ok := problem.Payload["hmcr"].(float64); ok {
		hmcr = v
	}

	par := harmonySearchPAR
	if v, ok := problem.Payload["par"].(float64); ok {
		par = v
	}

	bandwidth := harmonySearchBandwidth
	if v, ok := problem.Payload["bandwidth"].(float64); ok {
		bandwidth = v
	}

	maxSteps := harmonySearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	searchRange := harmonySearchRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}

	tolerance := harmonySearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	memory := make([][]float64, memorySize)
	for i := range memory {
		solution := make([]float64, dimensions)
		for d := range solution {
			solution[d] = (rand.Float64()*2 - 1) * searchRange
		}
		memory[i] = solution
	}

	best := memory[0]
	bestValue := fn.Evaluate(best)
	worstIndex := 0
	worstValue := bestValue
	for i, solution := range memory {
		value := fn.Evaluate(solution)
		if value < bestValue {
			bestValue = value
			best = solution
		}
		if value > worstValue {
			worstValue = value
			worstIndex = i
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		candidate := make([]float64, dimensions)
		for d := 0; d < dimensions; d++ {
			if rand.Float64() < hmcr {
				candidate[d] = memory[rand.IntN(memorySize)][d]
				if rand.Float64() < par {
					candidate[d] += (rand.Float64()*2 - 1) * bandwidth
				}
			} else {
				candidate[d] = (rand.Float64()*2 - 1) * searchRange
			}
		}

		candidateValue := fn.Evaluate(candidate)
		if candidateValue < worstValue {
			memory[worstIndex] = candidate
			if candidateValue < bestValue {
				bestValue = candidateValue
				best = candidate
			}

			worstIndex = 0
			worstValue = fn.Evaluate(memory[0])
			for i, solution := range memory {
				if value := fn.Evaluate(solution); value > worstValue {
					worstValue = value
					worstIndex = i
				}
			}
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
		Method:     "harmony_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
