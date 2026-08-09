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
	antColonyAnts        = 20
	antColonyEvaporation = 0.5
	antColonyMaxSteps    = 500
	antColonyRange       = 15.0
	antColonyTolerance   = 1e-6
)

// AntColony minimizuje konfigurisanu benchmark funkciju koristeći ACOR (Ant Colony Optimization za kontinualne domene):
// drži arhivu rešenja i uzorkuje nova iz Gausove raspodele centrirane na članovima arhive izabranim sa verovatnoćom
// ponderisanom po rangu, zadržavajući najbolji podskup veličine arhive posle svake runde
// problem.Point samo određuje dimenzionalnost
func AntColony(problem models.Problem) (result models.Result, err error) {

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

	ants := antColonyAnts
	if v, ok := problem.Payload["ants"].(float64); ok {
		ants = int(v)
	}

	evaporation := antColonyEvaporation
	if v, ok := problem.Payload["evaporation"].(float64); ok {
		evaporation = v
	}

	maxSteps := antColonyMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	searchRange := antColonyRange
	if v, ok := problem.Payload["range"].(float64); ok {
		searchRange = v
	}

	tolerance := antColonyTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	archive := make([][]float64, ants)
	for i := range archive {
		solution := make([]float64, dimensions)
		for d := range solution {
			solution[d] = (rand.Float64()*2 - 1) * searchRange
		}
		archive[i] = solution
	}
	sortByFitness(fn, archive)

	steps := 0
	for ; steps < maxSteps && fn.Evaluate(archive[0]) >= tolerance; steps++ {

		sigma := make([]float64, dimensions)
		for d := 0; d < dimensions; d++ {
			mean := 0.0
			for _, solution := range archive {
				mean += solution[d]
			}
			mean /= float64(len(archive))

			variance := 0.0
			for _, solution := range archive {
				diff := solution[d] - mean
				variance += diff * diff
			}
			variance /= float64(len(archive))

			sigma[d] = evaporation * math.Sqrt(variance)
		}

		offspring := make([][]float64, ants)
		for i := 0; i < ants; i++ {
			guide := archive[weightedIndex(len(archive))]

			solution := make([]float64, dimensions)
			for d := range solution {
				solution[d] = guide[d] + rand.NormFloat64()*sigma[d]
			}
			offspring[i] = solution
		}

		archive = append(archive, offspring...)
		sortByFitness(fn, archive)
		archive = archive[:ants]

		if includeTrajectory {
			trajectory = recordTrajectory(trajectory, steps, archive[0], fn.Evaluate(archive[0]), false)
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, archive[0], fn.Evaluate(archive[0]), true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "ant_colony",
		Point:      archive[0],
		Value:      fn.Evaluate(archive[0]),
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func sortByFitness(fn functions.BenchmarkFunction, solutions [][]float64) {
	sort.Slice(solutions, func(i, j int) bool {
		return fn.Evaluate(solutions[i]) < fn.Evaluate(solutions[j])
	})
}

// weightedIndex vraća nasumičan indeks ponderisan po rangu u sortiranom
// (najbolji-prvi) nizu date veličine, favorizujući niže (bolje) rangove.
func weightedIndex(size int) int {

	weights := make([]float64, size)
	totalWeight := 0.0
	for i := range weights {
		weights[i] = 1.0 / float64(i+1)
		totalWeight += weights[i]
	}

	target := rand.Float64() * totalWeight
	cumulative := 0.0
	for i, weight := range weights {
		cumulative += weight
		if target <= cumulative {
			return i
		}
	}

	return size - 1
}
