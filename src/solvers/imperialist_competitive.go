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
	imperialistCompetitivePopulation = 20
	imperialistCompetitiveMaxSteps   = 500
	imperialistCompetitiveNumEmpires = 4
	imperialistCompetitiveTolerance  = 1e-6
	imperialistCompetitiveRangeLow   = -5.0
	imperialistCompetitiveRangeHigh  = 5.0
)

// ImperialistCompetitive minimizuje konfigurisanu benchmark funkciju koristeći Imperialist Competitive Algorithm:
// kolonije asimiliraju se ka svojoj dodeljenoj imperiji i preuzimaju njeno mesto ako je nadmaše, dok najslabija
// imperija u svakom koraku gubi jednu nasumičnu koloniju u korist najjače
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func ImperialistCompetitive(problem models.Problem) (result models.Result, err error) {

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

	population := imperialistCompetitivePopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := imperialistCompetitiveMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	numEmpires := imperialistCompetitiveNumEmpires
	if v, ok := problem.Payload["num_empires"].(float64); ok {
		numEmpires = int(v)
	}
	if numEmpires < 1 {
		numEmpires = 1
	}
	if numEmpires > population-1 {
		numEmpires = population - 1
	}
	if numEmpires < 1 {
		numEmpires = 1
	}

	tolerance := imperialistCompetitiveTolerance
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
			individual[d] = imperialistCompetitiveRangeLow + rand.Float64()*(imperialistCompetitiveRangeHigh-imperialistCompetitiveRangeLow)
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, v := range values {
		if v < bestValue {
			bestValue = v
			best = append([]float64(nil), individuals[i]...)
		}
	}

	order := make([]int, population)
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return values[order[a]] < values[order[b]]
	})

	empireIdx := append([]int(nil), order[:numEmpires]...)
	colonyIdx := append([]int(nil), order[numEmpires:]...)

	// kolonije se raspoređuju na imperije rulet-selekcijom ponderisanom rangom snage — bolje rangirana
	// imperija (manji indeks u empireIdx, pošto je izvedena iz sortiranog niza) dobija više kolonija
	colonyToEmpire := make([]int, len(colonyIdx))
	for k := range colonyToEmpire {
		colonyToEmpire[k] = weightedIndex(len(empireIdx))
	}

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		for k, cIdx := range colonyIdx {
			eIdx := empireIdx[colonyToEmpire[k]]
			for d := range individuals[cIdx] {
				pull := rand.Float64() * 2.0 * (individuals[eIdx][d] - individuals[cIdx][d]) * (1 - float64(steps)/float64(maxSteps))
				noise := (rand.Float64()*2 - 1) * 0.3
				individuals[cIdx][d] = clamp(individuals[cIdx][d]+pull+noise, imperialistCompetitiveRangeLow, imperialistCompetitiveRangeHigh)
			}
			values[cIdx] = fn.Evaluate(individuals[cIdx])

			if values[cIdx] < values[eIdx] {
				individuals[cIdx], individuals[eIdx] = individuals[eIdx], individuals[cIdx]
				values[cIdx], values[eIdx] = values[eIdx], values[cIdx]
			}
		}

		// imperije same po sebi takođe moraju imati putanju ažuriranja — fina lokalna pretraga umesto da
		// ostanu zamrznute dok se takmičenje odvija samo oko kolonija
		for _, eIdx := range empireIdx {
			for d := range individuals[eIdx] {
				individuals[eIdx][d] = clamp(individuals[eIdx][d]+(rand.Float64()*2-1)*0.05, imperialistCompetitiveRangeLow, imperialistCompetitiveRangeHigh)
			}
			values[eIdx] = fn.Evaluate(individuals[eIdx])
		}

		weakestE, strongestE := 0, 0
		for e := 1; e < len(empireIdx); e++ {
			if values[empireIdx[e]] > values[empireIdx[weakestE]] {
				weakestE = e
			}
			if values[empireIdx[e]] < values[empireIdx[strongestE]] {
				strongestE = e
			}
		}
		if weakestE != strongestE {
			candidates := make([]int, 0)
			for k, e := range colonyToEmpire {
				if e == weakestE {
					candidates = append(candidates, k)
				}
			}
			if len(candidates) > 0 {
				colonyToEmpire[candidates[rand.IntN(len(candidates))]] = strongestE
			}
		}

		for i, v := range values {
			if v < bestValue {
				bestValue = v
				best = append([]float64(nil), individuals[i]...)
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
		Method:     "imperialist_competitive",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
