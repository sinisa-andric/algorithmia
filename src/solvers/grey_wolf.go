package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math/rand/v2"
)

const (
	greyWolfWolves    = 20
	greyWolfMaxSteps  = 500
	greyWolfTolerance = 1e-6
	greyWolfSpread    = 10.0
)

// GreyWolf minimizuje sphere funkciju koristeći Grey Wolf Optimizer: čopor vode njegova tri najbolja člana (alfa,
// beta, delta) uz koeficijent eksploracije koji linearno opada tokom izvršavanja
// problem.Point inicijalizuje čopor i određuje njegovu dimenzionalnost
func GreyWolf(problem models.Problem) (result models.Result, err error) {

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

	wolves := greyWolfWolves
	if v, ok := problem.Payload["wolves"].(float64); ok {
		wolves = int(v)
	}
	if wolves < 3 {
		wolves = 3
	}

	maxSteps := greyWolfMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := greyWolfTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)

	pack := make([][]float64, wolves)
	for i := range pack {
		wolf := make([]float64, dimensions)
		for d := range wolf {
			wolf[d] = problem.Point[d] + (rand.Float64()*2-1)*greyWolfSpread
		}
		pack[i] = wolf
	}

	sortByFitness(fn, pack)
	alpha, beta, delta := pack[0], pack[1], pack[2]

	best := append([]float64(nil), alpha...)
	bestValue := fn.Evaluate(alpha)

	steps := 0
	for ; steps < maxSteps && bestValue >= tolerance; steps++ {

		a := 2.0 - 2.0*float64(steps)/float64(maxSteps)

		for _, wolf := range pack {
			for d := 0; d < dimensions; d++ {
				a1 := a * (2*rand.Float64() - 1)
				c1 := 2 * rand.Float64()
				x1 := alpha[d] - a1*(c1*alpha[d]-wolf[d])

				a2 := a * (2*rand.Float64() - 1)
				c2 := 2 * rand.Float64()
				x2 := beta[d] - a2*(c2*beta[d]-wolf[d])

				a3 := a * (2*rand.Float64() - 1)
				c3 := 2 * rand.Float64()
				x3 := delta[d] - a3*(c3*delta[d]-wolf[d])

				wolf[d] = (x1 + x2 + x3) / 3
			}
		}

		sortByFitness(fn, pack)
		alpha, beta, delta = pack[0], pack[1], pack[2]

		if value := fn.Evaluate(alpha); value < bestValue {
			bestValue = value
			best = append([]float64(nil), alpha...)
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
		Method:     "grey_wolf",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}
