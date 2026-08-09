package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	differentialEvolutionPopulation = 20
	differentialEvolutionF          = 0.8
	differentialEvolutionCR         = 0.9
	differentialEvolutionMaxSteps   = 1000
	differentialEvolutionTolerance  = 1e-6
	differentialEvolutionSpread     = 10.0
)

// DifferentialEvolution minimizuje sphere funkciju koristeći diferencijalnu evoluciju (rand/1/bin): svaka jedinka se
// ukršta sa mutantom dobijenim od tri nasumično izabrane jedinke iz populacije
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func DifferentialEvolution(problem models.Problem) (result models.Result, err error) {

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

	population := differentialEvolutionPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	f := differentialEvolutionF
	if v, ok := problem.Payload["F"].(float64); ok {
		f = v
	}

	cr := differentialEvolutionCR
	if v, ok := problem.Payload["CR"].(float64); ok {
		cr = v
	}

	maxSteps := differentialEvolutionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := differentialEvolutionTolerance
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
			individual[d] = problem.Point[d] + (rand.Float64()*2-1)*differentialEvolutionSpread
		}
		individuals[i] = individual
		values[i] = fn.Evaluate(individual)
	}

	best := append([]float64(nil), individuals[0]...)
	bestValue := values[0]
	for i, value := range values {
		if value < bestValue {
			bestValue = value
			best = append([]float64(nil), individuals[i]...)
		}
	}

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		for i := 0; i < population; i++ {
			r1, r2, r3 := distinctIndices(population, i)

			trial := make([]float64, dimensions)
			for d := 0; d < dimensions; d++ {
				mutant := individuals[r1][d] + f*(individuals[r2][d]-individuals[r3][d])

				if rand.Float64() < cr {
					trial[d] = mutant
				} else {
					trial[d] = individuals[i][d]
				}
			}

			trialValue := fn.Evaluate(trial)
			if trialValue < values[i] {
				individuals[i] = trial
				values[i] = trialValue

				if trialValue < bestValue {
					bestValue = trialValue
					best = append([]float64(nil), trial...)
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
		Method:     "differential_evolution",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

func distinctIndices(population, exclude int) (int, int, int) {

	pick := func(taken map[int]bool) int {
		for {
			i := rand.IntN(population)
			if !taken[i] {
				return i
			}
		}
	}

	taken := map[int]bool{exclude: true}

	r1 := pick(taken)
	taken[r1] = true

	r2 := pick(taken)
	taken[r2] = true

	r3 := pick(taken)

	return r1, r2, r3
}
