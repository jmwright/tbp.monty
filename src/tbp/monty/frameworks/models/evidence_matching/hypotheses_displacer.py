# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from tbp.monty.frameworks.models.evidence_matching.channels import (
    all_usable_input_channels,
    is_null_channel,
)
from tbp.monty.frameworks.models.evidence_matching.feature_evidence.scorer import (
    FeatureEvidenceScorer,
)
from tbp.monty.frameworks.models.evidence_matching.graph_memory import (
    EvidenceGraphMemory,
)
from tbp.monty.frameworks.models.evidence_matching.hypotheses import Hypotheses
from tbp.monty.frameworks.utils.graph_matching_utils import (
    get_custom_distances,
    get_relevant_curvature,
)
from tbp.monty.frameworks.utils.spatial_arithmetics import (
    get_angles_for_all_hypotheses,
    rotate_pose_dependent_features,
)

logger = logging.getLogger(__name__)

MIN_EVIDENCE = -1
MAX_EVIDENCE = 2
EVIDENCE_RANGE = MAX_EVIDENCE - MIN_EVIDENCE


@dataclass
class HypothesisDisplacerTelemetry:
    mlh_prediction_error: float | None


class HypothesesDisplacer(Protocol):
    def displace_hypotheses(
        self,
        displacement: npt.NDArray[np.float64],
        hypotheses: Hypotheses,
    ) -> Hypotheses:
        """Displace hypothesis locations using the sensed sensor displacement.

        Each hypothesis location is updated by the pose-rotated displacement.
        Evidence, poses, and possible flags are left unchanged.

        Args:
            displacement: Displacement of the sensor (in common RF).
            hypotheses: Hypotheses to displace.

        Returns:
            Hypotheses with displaced locations.
        """
        ...

    def compute_evidence(
        self,
        features: dict[str, dict],
        evidence_update_threshold: float,
        graph_id: str,
        hypotheses: Hypotheses,
    ) -> tuple[Hypotheses, HypothesisDisplacerTelemetry]:
        """Updates evidence by comparing sensed features to features in the model.

        Uses the hypothesis locations as search locations for comparing features from
        all available input channels. Per-channel evidence is summed and applied as a
        weighted update. Assumes hypotheses have already been displaced for this step.

        Args:
            features: All-channel input features, keyed by channel name.
            evidence_update_threshold: Evidence update threshold.
            graph_id: The ID of the current graph.
            hypotheses: Hypotheses to compute evidence for.

        Returns:
            Hypotheses with computed evidence and telemetry.
        """
        ...


class DefaultHypothesesDisplacer:
    def __init__(
        self,
        feature_weights: dict,
        graph_memory: EvidenceGraphMemory,
        max_match_distance: float,
        feature_evidence_scorer: FeatureEvidenceScorer,
        max_nneighbors: int = 3,
        past_weight: float = 1,
        present_weight: float = 1,
        off_object_contradiction: float = 0.0,
        off_object_ray_carve: bool = False,
        off_object_ray_incidence: float = 0.5,
        off_object_coverage_normalised: bool = False,
    ):
        """Initializes the DefaultHypothesesDisplacer.

        Args:
            feature_weights: How much should each feature be weighted when
                calculating the evidence update for hypothesis. Weights are stored in a
                dictionary with keys corresponding to features (same as keys in
                tolerances).
            graph_memory: The graph memory to read graphs from.
            max_match_distance: Maximum distance of a tested and stored location
                to be matched.
            feature_evidence_scorer: Scorer that calculates evidence for all nodes in
                an object model's graph for a given channel.
            max_nneighbors: Maximum number of nearest neighbors to consider in the
                radius of a hypothesis for calculating the evidence. Defaults to 3.
            past_weight: How much should the evidence accumulated so far be
                weighted when combined with the evidence from the most recent
                observation. Defaults to 1.
            present_weight: How much should the current evidence be weighted
                when added to the previous evidence. If past_weight and present_weight
                add up to 1, the evidence is bounded and can't grow infinitely. Defaults
                to 1.
                NOTE: right now this doesn't give as good performance as with unbounded
                evidence since we don't keep a full history of what we saw. With a more
                efficient policy and better parameters that may be possible to use
                though and could help when moving from one object to another and to
                generally make setting thresholds etc. more intuitive.
            off_object_contradiction: Evidence subtracted from a hypothesis that
                is in model at a step where the sensor found no surface - it
                predicted a surface that is not there. Confirmation is no change
                rather than positive evidence, so this is the only value the null
                path applies. Defaults to 0.0, which makes the path inert.
            off_object_ray_incidence: How squarely a ray must meet the surface it
                strikes, as |ray . normal|, for the strike to count. 0 accepts a ray
                running along the surface and 1 demands head-on. Grazing strikes are
                what a distance test gets wrong near the silhouette, and rejecting
                them is what keeps the contradiction off the correct hypothesis.
                Ignored unless off_object_ray_carve is set. Defaults to 0.5,
                measured on the simulated glass episode: it strikes the correct
                object on 2 of 266 off-object rays against the point test's 6, the
                wrong one on 25 against 18, and still detects 97.6% of the rays that
                genuinely cross a surface. Raising it to 0.6 avoids one more false
                strike and starts missing real ones; lowering it to 0.4 more than
                doubles them. 0.3 discriminates slightly better at four times the false
                strikes.
            off_object_coverage_normalised: Whether the contradiction is a
                fraction of the hypothesis's own accumulated evidence rather than a
                fixed amount. A constant penalty is additive while positive support
                is not: on the simulated glass, ~300 on-object observations of a
                781-node graph swamp the handful that look where the handle should
                be, and `rig_mug` finishes above the termination band in all twelve
                azimuth bins by best *and* median. Scaling by the fraction of the
                hypothesis's own graph that a ray passed through and found empty
                makes the penalty proportional rather than absolute, so being shown
                that x% of your predicted surface is missing costs x% of your
                support however much support you have. Ignored unless
                off_object_ray_carve is set, which is what supplies the swept nodes.
                Defaults to False, which reproduces the fixed penalty exactly.
            off_object_ray_carve: Whether a null observation is tested as a ray
                rather than as a point. A void pixel does not assert "no surface at
                this depth", it asserts "no surface anywhere along this ray", so a
                hypothesis is contradicted wherever it predicts a surface the ray
                passes through. Testing the substituted point instead confines
                contradiction to whatever depth the substitution happened to pick,
                which leaves poses that are in plain view uncontradicted purely
                because their surface sits at a different depth. The result stays
                binary - one penalty for a hypothesis the ray passes through,
                regardless of how much of it does - so the evidence subtracted per
                step does not scale with object size or node density. Defaults to
                False, which keeps the point test.
        """
        self.feature_weights = feature_weights
        self.graph_memory = graph_memory
        self.max_match_distance = max_match_distance
        self.max_nneighbors = max_nneighbors
        self.past_weight = past_weight
        self.present_weight = present_weight
        self.off_object_contradiction = off_object_contradiction
        self.off_object_ray_carve = off_object_ray_carve
        self.off_object_ray_incidence = off_object_ray_incidence
        self.off_object_coverage_normalised = off_object_coverage_normalised
        self._ray_tolerances: dict[tuple[str, str], float] = {}
        self._feature_evidence_scorer = feature_evidence_scorer

    def displace_hypotheses(
        self,
        displacement: npt.NDArray[np.float64],
        hypotheses: Hypotheses,
    ) -> Hypotheses:
        # Have to do this for all hypotheses so we don't lose the path information
        # https://docs.thousandbrains.org/docs/glossary#path-integration
        rotated_displacements = hypotheses.poses.dot(displacement)
        search_locations = hypotheses.locations + rotated_displacements
        return Hypotheses(
            evidence=hypotheses.evidence,
            locations=search_locations,
            poses=hypotheses.poses,
            possible=hypotheses.possible,
        )

    def compute_evidence(
        self,
        features: dict[str, dict],
        evidence_update_threshold: float,
        graph_id: str,
        hypotheses: Hypotheses,
    ) -> tuple[Hypotheses, HypothesisDisplacerTelemetry]:
        search_locations = hypotheses.locations

        # Get indices of hypotheses with evidence > threshold
        hyp_idxs_to_test = np.where(hypotheses.evidence >= evidence_update_threshold)[0]
        num_hypotheses_to_test = hyp_idxs_to_test.shape[0]
        if num_hypotheses_to_test > 0:
            logger.info(
                f"Testing {num_hypotheses_to_test} out of "
                f"{hypotheses.count} hypotheses for {graph_id} "
                f"(evidence > {evidence_update_threshold})"
            )

            # Loop over channels, sum evidence
            input_channels = all_usable_input_channels(
                features, self.graph_memory.get_input_channels_in_graph(graph_id)
            )
            total_evidence_to_add = np.zeros_like(hypotheses.evidence)
            for channel in input_channels:
                new_evidence = self._calculate_evidence_for_new_locations(
                    graph_id=graph_id,
                    input_channel=channel,
                    search_locations=search_locations[hyp_idxs_to_test],
                    channel_possible_poses=hypotheses.poses[hyp_idxs_to_test],
                    channel_features=features[channel],
                    hypothesis_evidence=hypotheses.evidence[hyp_idxs_to_test],
                )
                min_update = np.clip(np.min(new_evidence), 0, np.inf)

                channel_evidence = np.ones_like(hypotheses.evidence) * min_update
                channel_evidence[hyp_idxs_to_test] = new_evidence
                total_evidence_to_add += channel_evidence

            # Prediction error from summed evidence
            mlh_index = np.argmax(hypotheses.evidence)
            evidence_for_mlh = total_evidence_to_add[mlh_index]

            # Each channel contributes evidence in range [MIN_EVIDENCE, MAX_EVIDENCE].
            # With C channels the summed range is [MIN_EVIDENCE * C, MAX_EVIDENCE * C].
            # We map to [1, 0] (inverted, since high evidence = low prediction error)
            # by negating, shifting by (MAX_EVIDENCE * C), and dividing by
            # (EVIDENCE_RANGE * C).
            # For C=1 and evidence in the range [-1, 2], this maps
            # [-1, 2] -> [1, 0] (e.g. evidence -1 -> error 1, evidence 2 -> error 0).
            num_channels = len(input_channels)
            mlh_prediction_error = (MAX_EVIDENCE * num_channels - evidence_for_mlh) / (
                EVIDENCE_RANGE * num_channels
            )

            # If past and present weight add up to 1, equivalent to
            # np.average and evidence will be bound to [-C, 2C] where C is the
            # number of channels. Otherwise it keeps growing.
            evidence = (
                hypotheses.evidence * self.past_weight
                + total_evidence_to_add * self.present_weight
            )
        else:
            evidence = hypotheses.evidence
            # If we haven't moved yet, there is no prediction, and thus no error
            mlh_prediction_error = None

        return Hypotheses(
            evidence=evidence,
            locations=search_locations,
            poses=hypotheses.poses,
            possible=hypotheses.possible,
        ), HypothesisDisplacerTelemetry(mlh_prediction_error=mlh_prediction_error)

    def _ray_tolerance(self, graph, graph_id, input_channel, nodes):
        """How close a ray must pass to a node to count as striking the surface.

        Not `max_match_distance`. That is a matching tolerance covering pose error,
        and using it here inflates the model by 10 mm in every direction, so a ray
        grazing just outside the silhouette strikes the object it is passing - which
        aims disconfirmation at the *correct* hypothesis. Measured on the simulated
        glass episode, the true pose was struck on 28 of 266 off-object steps at
        that tolerance against 4 at this one.

        Sized from the nodes instead. Measured on the simulated glass episode, the
        furthest any ray that genuinely crossed the surface fell from a stored node
        was 1.21 times the median spacing, so 1.25 times it is the radius at which
        real crossings are all detected. Half the spacing was tried first and is too
        tight - it detects 46% of genuine crossings and discriminates no better than
        the point test it replaces. It is measured per graph because the graphs
        differ: about 3.9 mm between nodes on the mug and the glass, 1.4 mm on the
        block.

        Returns:
            The tolerance in metres, cached per graph and input channel.
        """
        key = (graph_id, input_channel)

        if key not in self._ray_tolerances:
            # k=2 because the first neighbour of a node is itself, at zero distance.
            spacing = np.asarray(
                graph.find_nearest_neighbors(
                    nodes, num_neighbors=2, return_distance=True
                )
            )[:, 1]
            self._ray_tolerances[key] = 1.25 * float(np.median(spacing))

        return self._ray_tolerances[key]

    def _calculate_evidence_for_new_locations(
        self,
        graph_id: str,
        input_channel: str,
        search_locations: np.ndarray,
        channel_possible_poses: np.ndarray,
        channel_features: dict,
        hypothesis_evidence: np.ndarray | None = None,
    ):
        """Use search locations, sensed features and graph model to calculate evidence.

        First, the search locations are used to find the nearest nodes in the graph
        model. Then we calculate the error between the stored pose features and the
        sensed ones. Additionally we look at whether the non-pose features match at the
        neighboring nodes. Everything is weighted by the nodes distance from the search
        location.
        If there are no nodes in the search radius (max_match_distance), evidence = -1.

        We do this for every incoming input channel and its features if they are stored
        in the graph and take the average over the evidence from all input channels.

        Returns:
            The location evidence.
        """
        logger.debug(
            f"Calculating evidence for {graph_id} using input from {input_channel}"
        )

        if is_null_channel(channel_features):
            direction = channel_features.get("ray_direction")
            graph = self.graph_memory.get_graph(graph_id, input_channel)

            if not self.off_object_ray_carve or direction is None:
                dists = graph.find_nearest_neighbors(
                    search_locations, num_neighbors=1, return_distance=True
                )
                # Contradicted where the hypothesis predicts a surface the sensor
                # did not find. Confirmed as no change, not positive evidence.
                in_model = np.asarray(dists) <= self.max_match_distance
                return np.where(in_model, -self.off_object_contradiction, 0.0)

            nodes = self.graph_memory.get_locations_in_graph(graph_id, input_channel)
            center = nodes.mean(axis=0)
            radius = float(np.linalg.norm(nodes - center, axis=1).max())
            tolerance = self._ray_tolerance(graph, graph_id, input_channel, nodes)

            # The ray in each hypothesis's own frame. `poses` maps sensor-frame
            # vectors into the model frame - measured, not assumed.
            rays = np.einsum("hij,j->hi", channel_possible_poses, direction)
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)

            # Distance along each ray to the point nearest the model's centre. The
            # window has to be centred here rather than on the search location: a
            # null observation's search location sits off the object by
            # construction, so a +-radius window about it need not reach the model
            # at all. Centred on the closest approach it always covers the chord,
            # which is at most 2*radius long, however far away the point is. The
            # step must not exceed the tolerance or a thin feature is stepped over.
            closest = np.einsum(
                "hi,hi->h", center[None, :] - search_locations, rays
            )
            span = np.arange(-radius, radius + tolerance, tolerance)
            along = closest[:, None] + span[None, :]

            samples = (
                search_locations[:, None, :] + along[:, :, None] * rays[:, None, :]
            )
            # Indices rather than distances, because the normal at the struck node
            # is needed too and one query gives both.
            ids = np.asarray(graph.find_nearest_neighbors(
                samples.reshape(-1, 3), num_neighbors=1, return_distance=False
            )).reshape(len(search_locations), -1)
            struck = nodes[ids]
            dists = np.linalg.norm(samples - struck, axis=2)

            # Distance alone cannot separate a crossing from a graze. At the
            # silhouette the surface is tangent to the view, so a ray just outside
            # passes as close to a stored node as one just inside - measured, the
            # median on-object ray passes 1.55 mm from a node and the closest
            # off-object ray 1.17 mm, so the two distributions overlap and no
            # threshold divides them. Incidence is a second, independent axis: a ray
            # that crosses the surface meets it near normal, while one grazing the
            # silhouette runs along it. Requiring both took the correct hypothesis
            # from 28 false strikes to 1, while striking the wrong object more often
            # than the point test did.
            normals = np.asarray(graph.norm, dtype=float)
            normals = normals / np.maximum(
                np.linalg.norm(normals, axis=1, keepdims=True), 1e-12
            )
            incidence = np.abs(np.einsum("hsi,hi->hs", normals[ids], rays))

            struck_here = (
                (dists <= tolerance) & (incidence >= self.off_object_ray_incidence)
            )
            hit = struck_here.any(axis=1)

            if not self.off_object_coverage_normalised or hypothesis_evidence is None:
                return np.where(hit, -self.off_object_contradiction, 0.0)

            # How much of this hypothesis's own model the ray passed through and
            # found empty, as a fraction of the model. Counted in distinct nodes
            # rather than samples, because the sample step is the tolerance and a
            # ray running along a surface would otherwise score a long chord off a
            # handful of nodes.
            coverage = np.array(
                [
                    len(np.unique(ids[h][struck_here[h]])) / len(nodes)
                    for h in range(len(search_locations))
                ]
            )

            # Proportional, not absolute. `past_weight` and `present_weight` are
            # both 1 in every config here, so the update is additive and this makes
            # the step multiplicative: evidence *= (1 - contradiction * coverage).
            # Clipped at zero so a hypothesis already in negative evidence is not
            # rewarded for being contradicted, and capped so one observation cannot
            # invert the sign of the evidence it is scaling.
            support = np.clip(hypothesis_evidence, 0.0, np.inf)
            scale = np.clip(self.off_object_contradiction * coverage, 0.0, 1.0)

            return -support * scale

        pose_transformed_features = rotate_pose_dependent_features(
            channel_features,
            channel_possible_poses,
        )
        # Get up to max_nneighbors nearest nodes to search locations.
        channel_locations = self.graph_memory.get_locations_in_graph(
            graph_id, input_channel
        )
        num_neighbors = min(self.max_nneighbors, channel_locations.shape[0])
        nearest_node_ids = self.graph_memory.get_graph(
            graph_id, input_channel
        ).find_nearest_neighbors(
            search_locations,
            num_neighbors=num_neighbors,
        )
        if num_neighbors == 1:
            nearest_node_ids = np.expand_dims(nearest_node_ids, axis=1)

        nearest_node_locs = channel_locations[nearest_node_ids]
        max_abs_curvature = get_relevant_curvature(channel_features)
        custom_nearest_node_dists = get_custom_distances(
            nearest_node_locs,
            search_locations,
            pose_transformed_features["pose_vectors"][:, 0],
            max_abs_curvature,
        )
        # shape=(H, K)
        node_distance_weights = self._get_node_distance_weights(
            custom_nearest_node_dists
        )
        # Get IDs where custom_nearest_node_dists > max_match_distance
        mask = node_distance_weights <= 0

        new_pos_features = self.graph_memory.get_features_at_node(
            graph_id,
            input_channel,
            nearest_node_ids,
            feature_keys=["pose_vectors", "pose_fully_defined"],
        )
        # Calculate the pose error for each hypothesis
        # shape=(H, K)
        radius_evidence = self._get_pose_evidence_matrix(
            pose_transformed_features,
            new_pos_features,
            input_channel,
            node_distance_weights,
        )
        # Set the evidences which are too far away to -1
        radius_evidence[mask] = -1
        # If a node is too far away, weight the negative evidence fully (*1). This
        # only comes into play if there are no nearby nodes in the radius, then we
        # want an evidence of -1 for this hypothesis.
        # NOTE: Currently we don't weight the evidence by distance so this doesn't
        # matter.
        node_distance_weights[mask] = 1

        node_feature_evidence = self._feature_evidence_scorer(
            graph_id=graph_id,
            input_channel=input_channel,
            query_features=channel_features,
        )
        hypothesis_radius_feature_evidence = node_feature_evidence[nearest_node_ids]
        # Set feature evidence of nearest neighbors that are too far away to 0
        hypothesis_radius_feature_evidence[mask] = 0
        # Take the maximum feature evidence out of the nearest neighbors in the
        # search radius and weighted by its distance to the search location.
        # Evidence will be in [0, 1] and is only 1 if all features match
        # perfectly and the node is at the search location.
        radius_evidence = radius_evidence + hypothesis_radius_feature_evidence

        # We take the maximum to be better able to deal with parts of the model where
        # features change quickly and we may have noisy location information. This way
        # we check if we can find a good match of pose features within the search
        # radius. It doesn't matter if there are also points stored nearby in the model
        # that are not a good match.
        # Removing the comment weights the evidence by the nodes distance from the
        # search location. However, empirically this did not seem to help.
        # shape=(H,)
        return np.max(
            radius_evidence,  # * node_distance_weights,
            axis=1,
        )

    def _get_node_distance_weights(self, distances):
        return (self.max_match_distance - distances) / self.max_match_distance

    def _get_pose_evidence_matrix(
        self,
        query_features,
        node_features,
        input_channel,
        node_distance_weights,
    ):
        """Get angle mismatch error of the three pose features for multiple points.

        Args:
            query_features: Observed features.
            node_features: Features at nodes that are being tested.
            input_channel: Input channel for which we want to calculate the
                pose evidence. These are all input channels that are received at the
                current time step and are also stored in the graph.
            node_distance_weights: Weights for each nodes error (determined by
                distance to the search location). Currently not used, except for shape.

        Returns:
            The sum of angle evidence weighted by weights. In range [-1, 1].
        """
        # TODO S: simplify by looping over pose vectors
        evidences_shape = node_distance_weights.shape[:2]
        pose_evidence_weighted = np.zeros(evidences_shape)
        # TODO H: at higher level LMs we may want to look at all pose vectors.
        # Currently we skip the third since the second curv dir is always 90 degree
        # from the first.
        # Get angles between three pose features
        surface_normal_error = get_angles_for_all_hypotheses(
            # shape of node_features[input_channel]["pose_vectors"]: (nH, knn, 9)
            node_features["pose_vectors"][:, :, :3],
            query_features["pose_vectors"][:, 0],  # shape (nH, 3)
        )
        # Divide error by 2 so it is in range [0, pi/2]
        # Apply sin -> [0, 1]. Subtract 0.5 -> [-0.5, 0.5]
        # Negate the error to get evidence (lower error is higher evidence)
        surface_normal_evidence = -(np.sin(surface_normal_error / 2) - 0.5)
        surface_normal_weight = self.feature_weights[input_channel]["pose_vectors"][0]
        # If curvatures are same the directions are meaningless
        #  -> set curvature angle error to zero.
        if not query_features["pose_fully_defined"]:
            cd1_weight = 0
            # Only calculate curv dir angle if sensed curv dirs are meaningful
            cd1_evidence = np.zeros(surface_normal_error.shape)
            # TODO: Test whether we should double the SN evidence if no
            # curvatures are sensed and pose_fully_defined == False at node.
            # i.e. move use_cd from else block and set
            # surface_normal_evidence[np.logical_not(use_cd)] *= 2 (see PR#446)
        else:
            cd1_weight = self.feature_weights[input_channel]["pose_vectors"][1]
            # Also check if curv dirs stored at node are meaningful
            use_cd = np.array(
                node_features["pose_fully_defined"][:, :, 0],
                dtype=bool,
            )
            cd1_angle = get_angles_for_all_hypotheses(
                node_features["pose_vectors"][:, :, 3:6],
                query_features["pose_vectors"][:, 1],
            )
            # Since curvature directions could be rotated 180 degrees we define the
            # error to be largest when the angle is pi/2 (90 deg) and angles 0 and
            # pi are equal. This means the angle error will be between 0 and pi/2.
            cd1_error = np.pi / 2 - np.abs(cd1_angle - np.pi / 2)
            # We then apply the same operations as on surface_normal error to get
            # cd1_evidence
            # in range [-0.5, 0.5]
            cd1_evidence = -(np.sin(cd1_error) - 0.5)
            # nodes where pc1==pc2 receive no cd evidence but twice the surface_normal
            # evidence
            # -> overall evidence can be in range [-1, 1]
            cd1_evidence = cd1_evidence * use_cd
        # weight angle errors by feature weights
        # if sensed pc1==pc2 cd1_weight==0 and overall evidence is in [-0.5, 0.5]
        # otherwise it is in [-1, 1].
        pose_evidence_weighted += (
            surface_normal_evidence * surface_normal_weight + cd1_evidence * cd1_weight
        )
        return pose_evidence_weighted
