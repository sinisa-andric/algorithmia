package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	cuckooSearchNests     = 20
	cuckooSearchPA        = 0.25
	cuckooSearchMaxSteps  = 1000
	cuckooSearchTolerance = 1e-6
	cuckooSearchSpread    = 10.0
	cuckooSearchLevyScale = 0.1
	cuckooSearchBound     = 15.0
	cuckooSearchLevyPower = 1.5
)

// CuckooSearch minimizuje sphere funkciju koristeći cuckoo search: svako gnezdo pravi Levy-flight korak ka trenutno
// najboljem, a deo najgorih gnezda se svake runde napušta i nasumično zamenjuje
// problem.Point inicijalizuje gnezda i određuje njihovu dimenzionalnost
func CuckooSearch(problem models.Problem) (result models.Result, err error) {

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

	nestCount := cuckooSearchNests
	if v, ok := problem.Payload["nests"].(float64); ok {
		nestCount = int(v)
	}

	pa := cuckooSearchPA
	if v, ok := problem.Payload["pa"].(float64); ok {
		pa = v
	}

	maxSteps := cuckooSearchMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := cuckooSearchTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}

	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	nests := make([][]float64, nestCount)
	for i := range nests {
		nest := make([]float64, dimensions)
		for d := range nest {
			nest[d] = problem.Point[d] + (rand.Float64()*2-1)*cuckooSearchSpread
		}
		nests[i] = nest
	}
	sortByFitness(fn, nests)

	best := append([]float64(nil), nests[0]...)
	bestValue := fn.Evaluate(best)

	abandon := int(pa * float64(nestCount))

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		for i, nest := range nests {
			newPos := make([]float64, dimensions)
			for d := range nest {
				levyStep := cuckooSearchLevyScale * randNorm() / math.Pow(math.Abs(randNorm()), 1/cuckooSearchLevyPower)
				newPos[d] = nest[d] + levyStep*(nest[d]-best[d])

				if newPos[d] > cuckooSearchBound {
					newPos[d] = cuckooSearchBound
				} else if newPos[d] < -cuckooSearchBound {
					newPos[d] = -cuckooSearchBound
				}
			}

			if fn.Evaluate(newPos) < fn.Evaluate(nest) {
				nests[i] = newPos
			}
		}

		sortByFitness(fn, nests)
		for i := nestCount - abandon; i < nestCount; i++ {
			nest := make([]float64, dimensions)
			for d := range nest {
				nest[d] = problem.Point[d] + (rand.Float64()*2-1)*cuckooSearchSpread
			}
			nests[i] = nest
		}

		sortByFitness(fn, nests)
		if value := fn.Evaluate(nests[0]); value < bestValue {
			bestValue = value
			best = append([]float64(nil), nests[0]...)
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
		Method:     "cuckoo_search",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
