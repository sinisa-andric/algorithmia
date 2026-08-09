package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	treeGrowthPopulation = 20
	treeGrowthMaxSteps   = 500
	treeGrowthTolerance  = 1e-6
	treeGrowthRangeLow   = -5.0
	treeGrowthRangeHigh  = 5.0
)

// TreeGrowth minimizuje konfigurisanu benchmark funkciju koristeći Tree Growth Algorithm: svako stablo se takmiči
// za svetlost prema sopstvenom kvalitetu u odnosu na najgoru jedinku u populaciji, pa bolje osvetljena stabla jače
// rastu ka najboljem rešenju dok slabije osvetljena granaju ka otvorenom (nasumičnom) prostoru
// problem.Point inicijalizuje populaciju stabala i određuje njenu dimenzionalnost
func TreeGrowth(problem models.Problem) (result models.Result, err error) {

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

	population := treeGrowthPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := treeGrowthMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := treeGrowthTolerance
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
			individual[d] = treeGrowthRangeLow + rand.Float64()*(treeGrowthRangeHigh-treeGrowthRangeLow)
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

		worstValue := values[0]
		for _, v := range values {
			if v > worstValue {
				worstValue = v
			}
		}

		for i := range individuals {
			lightCompetition := (worstValue - values[i]) / (worstValue - bestValue + 1e-10)
			for d := range individuals[i] {
				pull := lightCompetition * rand.Float64() * (best[d] - individuals[i][d])
				noise := (1 - lightCompetition) * (rand.Float64()*2 - 1) * 0.3
				individuals[i][d] = clamp(individuals[i][d]+pull+noise, treeGrowthRangeLow, treeGrowthRangeHigh)
			}
			values[i] = fn.Evaluate(individuals[i])
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
		Method:     "tree_growth",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
