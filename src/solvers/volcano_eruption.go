package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	volcanoEruptionPopulation = 20
	volcanoEruptionMaxSteps   = 500
	volcanoEruptionTolerance  = 1e-6
	volcanoEruptionRangeLow   = -5.0
	volcanoEruptionRangeHigh  = 5.0
)

// VolcanoEruption minimizuje konfigurisanu benchmark funkciju koristeći Volcano Eruption Algorithm: sa
// verovatnoćom koja opada tokom izvršavanja jedinka erumpira velikim skokom čija veličina raste sa udaljenošću od
// best-a, a inače prolazi kroz fazu mirovanja sa finom pretragom ka best-u
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func VolcanoEruption(problem models.Problem) (result models.Result, err error) {

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

	population := volcanoEruptionPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := volcanoEruptionMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := volcanoEruptionTolerance
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
			individual[d] = volcanoEruptionRangeLow + rand.Float64()*(volcanoEruptionRangeHigh-volcanoEruptionRangeLow)
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

		eruptionProb := 0.3 * (1 - float64(steps)/float64(maxSteps))

		for i := range individuals {
			if rand.Float64() < eruptionProb {
				diff := make([]float64, dimensions)
				for d := range diff {
					diff[d] = individuals[i][d] - best[d]
				}
				distance := norm(diff)
				jumpScale := 1 - 1/(1+distance)
				for d := range individuals[i] {
					individuals[i][d] = clamp(individuals[i][d]+(rand.Float64()*2-1)*2.0*jumpScale, volcanoEruptionRangeLow, volcanoEruptionRangeHigh)
				}
			} else {
				for d := range individuals[i] {
					pull := rand.Float64() * (best[d] - individuals[i][d]) * 0.3
					noise := (rand.Float64()*2 - 1) * 0.05
					individuals[i][d] = clamp(individuals[i][d]+pull+noise, volcanoEruptionRangeLow, volcanoEruptionRangeHigh)
				}
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
		Method:     "volcano_eruption",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
