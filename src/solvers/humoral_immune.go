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
	humoralPopulation  = 20
	humoralMaxSteps    = 500
	humoralPlasmaRatio = 0.2
	humoralTolerance   = 1e-6
	humoralRangeLow    = -5.0
	humoralRangeHigh   = 5.0
)

// HumoralImmune minimizuje konfigurisanu benchmark funkciju koristeći Humoral Immune Response Algorithm: najbolje
// rangirane plazma ćelije fino pretražuju sopstvenu okolinu amplitudom koja opada tokom izvršavanja, dok preostale
// ćelije hipermutiraju obrnuto proporcionalno rangu sa vezom ka najboljem pronađenom rešenju
// problem.Point inicijalizuje populaciju B-ćelija i određuje njenu dimenzionalnost
func HumoralImmune(problem models.Problem) (result models.Result, err error) {

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

	population := humoralPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := humoralMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	plasmaRatio := humoralPlasmaRatio
	if v, ok := problem.Payload["plasma_ratio"].(float64); ok {
		plasmaRatio = v
	}

	tolerance := humoralTolerance
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
			individual[d] = humoralRangeLow + rand.Float64()*(humoralRangeHigh-humoralRangeLow)
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

	maxNoImprove := max(50, maxSteps/20)
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		order := make([]int, population)
		for i := range order {
			order[i] = i
		}
		sort.Slice(order, func(a, b int) bool {
			return values[order[a]] < values[order[b]]
		})

		granica := int(math.Round(plasmaRatio * float64(population)))
		if granica > population {
			granica = population
		}

		for r := 0; r < granica; r++ {
			idx := order[r]
			for d := range individuals[idx] {
				amplitude := 0.15 * (1 - float64(steps)/float64(maxSteps))
				individuals[idx][d] = clamp(individuals[idx][d]+(rand.Float64()*2-1)*amplitude, humoralRangeLow, humoralRangeHigh)
			}
			values[idx] = fn.Evaluate(individuals[idx])
		}

		for r := granica; r < population; r++ {
			idx := order[r]
			for d := range individuals[idx] {
				pull := rand.Float64() * (best[d] - individuals[idx][d]) * 0.3
				individuals[idx][d] = clamp(individuals[idx][d]+(rand.Float64()*2-1)*0.6+pull, humoralRangeLow, humoralRangeHigh)
			}
			values[idx] = fn.Evaluate(individuals[idx])
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
		Method:     "humoral_immune",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
