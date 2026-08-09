package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
	"sort"
)

const (
	airsPopulation    = 20
	airsMaxSteps      = 500
	airsMemorySize    = 5
	airsMutationScale = 0.2
	airsTolerance     = 1e-6
	airsRangeLow      = -5.0
	airsRangeHigh     = 5.0
)

// Airs minimizuje konfigurisanu benchmark funkciju koristeći Artificial Immune Recognition System: ARB-ovi se
// pomeraju ka najbližoj memorijskoj ćeliji i nadmeću se za resurse (najslabiji po stimulaciji se zamenjuju
// nasumičnim novim), a memorijske ćelije se ažuriraju čim se pronađe bolji kandidat među ARB-ovima
// problem.Point inicijalizuje populaciju ARB-ova i određuje njenu dimenzionalnost
func Airs(problem models.Problem) (result models.Result, err error) {

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

	population := airsPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := airsMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	memorySize := airsMemorySize
	if v, ok := problem.Payload["memory_size"].(float64); ok {
		memorySize = int(v)
	}

	mutationScale := airsMutationScale
	if v, ok := problem.Payload["mutation_scale"].(float64); ok {
		mutationScale = v
	}

	tolerance := airsTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	individuals := make([][]float64, population)
	values := make([]float64, population)
	for i := range individuals {
		individual := make([]float64, dimensions)
		for d := range individual {
			individual[d] = airsRangeLow + rand.Float64()*(airsRangeHigh-airsRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	if memorySize > population {
		memorySize = population
	}

	order := make([]int, population)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return values[order[a]] < values[order[b]]
	})

	memoryCells := make([][]float64, memorySize)
	memoryValues := make([]float64, memorySize)
	for m := 0; m < memorySize; m++ {
		memoryCells[m] = append([]float64(nil), individuals[order[m]]...)
		memoryValues[m] = values[order[m]]
	}

	bestIdx := 0
	for i, v := range memoryValues {
		if v < memoryValues[bestIdx] {
			bestIdx = i
		}
	}
	best := append([]float64(nil), memoryCells[bestIdx]...)
	bestValue := memoryValues[bestIdx]

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		stimulation := make([]float64, population)
		for i := range individuals {
			nearestIdx := 0
			nearestDist := math.Inf(1)
			for m := range memoryCells {
				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = individuals[i][d] - memoryCells[m][d]
				}
				dist := norm(diff)
				if dist < nearestDist {
					nearestDist = dist
					nearestIdx = m
				}
			}
			stimulation[i] = (1 / (values[i] + 1e-10)) * (1 / (nearestDist + 1e-10))

			for d := range individuals[i] {
				pull := rand.Float64() * (memoryCells[nearestIdx][d] - individuals[i][d]) * mutationScale
				individuals[i][d] = clamp(individuals[i][d]+pull+(rand.Float64()*2-1)*0.1, airsRangeLow, airsRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
		}

		// resource competition: donjih 20% ARB-ova po stimulaciji se zamenjuje nasumičnim novim
		numReplace := int(0.2 * float64(population))
		if numReplace > 0 {
			stimOrder := make([]int, population)
			for i := range stimOrder {
				stimOrder[i] = i
			}
			sort.Slice(stimOrder, func(a, b int) bool {
				return stimulation[stimOrder[a]] < stimulation[stimOrder[b]]
			})
			for i := 0; i < numReplace && i < population; i++ {
				idx := stimOrder[i]
				individual := make([]float64, dimensions)
				for d := range individual {
					individual[d] = airsRangeLow + rand.Float64()*(airsRangeHigh-airsRangeLow)
				}
				individuals[idx] = individual
				values[idx] = fn.Evaluate(individual)
			}
		}

		// svaki ARB koji je bolji od trenutno najgore memorijske ćelije je zamenjuje, tako da memorijske
		// ćelije nikad ne ostaju zamrznute na inicijalnoj poziciji
		for i := range individuals {
			worstIdx := 0
			for m := 1; m < len(memoryValues); m++ {
				if memoryValues[m] > memoryValues[worstIdx] {
					worstIdx = m
				}
			}
			if values[i] < memoryValues[worstIdx] {
				memoryCells[worstIdx] = append([]float64(nil), individuals[i]...)
				memoryValues[worstIdx] = values[i]
			}
		}

		for _, v := range memoryValues {
			if v < bestValue {
				bestValue = v
			}
			if includeTrajectory {
				trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
			}
		}
		for i, v := range memoryValues {
			if v == bestValue {
				best = append([]float64(nil), memoryCells[i]...)
				break
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
		Method:     "airs",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
