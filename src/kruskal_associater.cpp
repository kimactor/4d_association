#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <numeric>
#include <algorithm>
#include "kruskal_associater.h"
#include "math_util.h"


KruskalAssociater::KruskalAssociater(const SkelType& type, const std::map<std::string, Camera>& cams)
	:Associater(type, cams) {

	const SkelDef& def = GetSkelDef(m_type);
	m_joint2paf.resize(def.jointSize);
	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++) {
		const auto jIdxPair = def.pafDict.col(pafIdx);
		m_joint2paf[jIdxPair.x()].emplace_back(pafIdx);
		m_joint2paf[jIdxPair.y()].emplace_back(pafIdx);
	}

	m_pafHier.resize(def.pafSize);
	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++)
		m_pafHier[pafIdx] = std::min(def.hierarchyMap[def.pafDict(0, pafIdx)], def.hierarchyMap[def.pafDict(1, pafIdx)]);
	m_pafHierSize = m_pafHier.maxCoeff();

	m_boneNodes.resize(def.pafSize, std::vector<std::vector<Eigen::Vector2i>>(m_cams.size()));
	m_boneEpiEdges.resize(def.pafSize, std::vector<std::vector<Eigen::MatrixXf>>(m_cams.size(), std::vector<Eigen::MatrixXf>(m_cams.size())));
	m_boneTempEdges.resize(def.pafSize, std::vector<Eigen::MatrixXf>(m_cams.size()));
}


void KruskalAssociater::CalcBoneNodes()
{
	const SkelDef& def = GetSkelDef(m_type);
#pragma omp parallel for
	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++) {
		const int jaIdx = def.pafDict(0, pafIdx);
		const int jbIdx = def.pafDict(1, pafIdx);
		for (int view = 0; view < m_cams.size(); view++) {
			m_boneNodes[pafIdx][view].clear();
			for (int jaCandiIdx = 0; jaCandiIdx < m_detections[view].joints[jaIdx].cols(); jaCandiIdx++)
				for (int jbCandiIdx = 0; jbCandiIdx < m_detections[view].joints[jbIdx].cols(); jbCandiIdx++)
					if (m_detections[view].pafs[pafIdx](jaCandiIdx, jbCandiIdx) > FLT_EPSILON)
						m_boneNodes[pafIdx][view].emplace_back(Eigen::Vector2i(jaCandiIdx, jbCandiIdx));
		}
	}
}


void KruskalAssociater::CalcBoneEpiEdges()
{
	const SkelDef& def = GetSkelDef(m_type);
#pragma omp parallel for
	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++) {
		const Eigen::Vector2i jIdxPair = def.pafDict.col(pafIdx).transpose();
		for (int viewA = 0; viewA < m_cams.size() - 1; viewA++) {
			for (int viewB = viewA + 1; viewB < m_cams.size(); viewB++) {
				Eigen::MatrixXf& epi = m_boneEpiEdges[pafIdx][viewA][viewB];
				const auto& nodesA = m_boneNodes[pafIdx][viewA];
				const auto& nodesB = m_boneNodes[pafIdx][viewB];
				epi.setConstant(nodesA.size(), nodesB.size(), -1.f);
				for (int boneAIdx = 0; boneAIdx < epi.rows(); boneAIdx++) {
					for (int boneBIdx = 0; boneBIdx < epi.cols(); boneBIdx++) {
						const Eigen::Vector2i& nodeA = nodesA[boneAIdx];
						const Eigen::Vector2i& nodeB = nodesB[boneBIdx];

						Eigen::Vector2f epiDist;
						//Eigen::Matrix<float, 3, 2> normals;
						for (int i = 0; i < 2; i++) {
							//normals.col(i) = m_jointRays[viewA][jIdxPair[i]].col(nodeA[i]).cross(
							//	m_jointRays[viewB][jIdxPair[i]].col(nodeB[i])).normalized();
							epiDist[i] = m_epiEdges[jIdxPair[i]][viewA][viewB](nodeA[i], nodeB[i]);
						}

						if (epiDist.minCoeff() < 0.f)
							continue;

						//const float cosine = fabsf(normals.col(0).dot(normals.col(1)));
						epi(boneAIdx, boneBIdx) = epiDist.mean();
					}
				}
				m_boneEpiEdges[pafIdx][viewB][viewA] = epi.transpose();
			}
		}
	}
}


void KruskalAssociater::CalcBoneTempEdges()
{
	const SkelDef& def = GetSkelDef(m_type);
#pragma omp parallel for
	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++) {
		const Eigen::Vector2i jIdxPair = def.pafDict.col(pafIdx).transpose();
		for (int view = 0; view < m_cams.size(); view++) {
			Eigen::MatrixXf& temp = m_boneTempEdges[pafIdx][view];
			const auto& nodes = m_boneNodes[pafIdx][view];
			temp.setConstant(m_skels3dPrev.size(), nodes.size(), -1.f);
			for (int pIdx = 0; pIdx < temp.rows(); pIdx++) {
				for (int jCandiIdx = 0; jCandiIdx < temp.cols(); jCandiIdx++) {
					const Eigen::Vector2i& node = nodes[jCandiIdx];
					Eigen::Vector2f tempDist;
					for (int i = 0; i < 2; i++)
						tempDist[i] = m_tempEdges[jIdxPair[i]][view](pIdx, node[i]);

					if (tempDist.minCoeff() > 0.f)
						temp(pIdx, jCandiIdx) = tempDist.mean();
				}
			}
		}
	}
}

inline void printAvailNodes(int id, const std::vector<std::vector<std::list<int>>>& availNodes)
{
	std::cout << "--------------" << std::endl;
	std::cout << "index: " << id <<  " AvaliNodes: " << std::endl;
	for (int index = 0; index < availNodes.size(); ++index)
	{
		for (int view = 0; view < availNodes[index].size(); ++view)
		{
			std::cout << "(";
			int list_id = 0;
			for (auto& p1: availNodes[index][view])
			{
				std::cout << p1;
				if (list_id != (availNodes[index][view].size() - 1))
				{
					std::cout << " ";
				}
				list_id++;
			}
			std::cout << ")";
		}
		std::cout << std::endl;
	}
	std::cout << "--------------" << std::endl;

}
   
         
void KruskalAssociater::EnumCliques(std::vector<BoneClique>& cliques)
{
	// enum cliques
	const SkelDef& def = GetSkelDef(m_type);

	std::vector<std::vector<BoneClique>> tmpCliques(def.pafSize);
//#pragma omp parallel for
	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++) {
		const auto jIdxPair = def.pafDict.col(pafIdx);
		const auto& nodes = m_boneNodes[pafIdx];
		// m_cams size ÿ����ͼ������һ�������Ѿ����ɵ���ά����Ǽ�
		Eigen::VectorXi pick = -Eigen::VectorXi::Ones(m_cams.size() + 1);
		std::vector<std::vector<std::list<int>>> availNodes(pick.size(), std::vector<std::list<int>>(pick.size()));

		int viewCnt = 0;
		int index = -1;
		while (true) {
			if (index >= 0 && pick[index] >= int(availNodes[index][index].size())) {
				std::cout << "current: " << index << " pick[index]: " << pick[index] << " availiNodes value: " << int(availNodes[index][index].size()) << std::endl;
				pick[index] = -1;

				if (--index < 0)
					break;
				pick[index]++;
				std::cout << "update Pick: " << pick.transpose() << std::endl;
			}
			else if (index == pick.size() - 1) {
				//��pick ���в�Ϊ-1 ��ֵ
				if (-pick.head(m_cams.size()).sum() != m_cams.size()) {
					BoneClique clique;
					clique.pafIdx = pafIdx;
					clique.proposal.setConstant(pick.size(), -1);
					for (int i = 0; i < pick.size(); i++)
						if (pick[i] != -1)
							clique.proposal[i] = *std::next(availNodes[i][i].begin(), pick[i]);

					CalcCliqueScore(clique);
					tmpCliques[pafIdx].emplace_back(clique);
					std::cout << "emplace back clique: " << index << std::endl;
				}
				pick[index]++;
				std::cout << "only execute index = 5 update Pick: " << pick.transpose() << std::endl;
			}
			else {
				index++;
				// update available nodes
				if (index == 0) {
					for (int view = 0; view < m_cams.size(); view++) {
						const auto& asgnMap = m_assignMap[view];
						for (int bone = 0; bone < nodes[view].size(); bone++) 
							availNodes[0][view].emplace_back(bone);
					}
					for (int pIdx = 0; pIdx < m_skels3dPrev.size(); pIdx++)
						availNodes[0].back().emplace_back(pIdx);

					std::cout << "set value index = 0" << std::endl;
					printAvailNodes(index, availNodes);

				}
				else {
					// epipolar constrain
					if (pick[index - 1] >= 0) {
						std::cout << "epipolar constrain " <<  index << std::endl;
						for (int view = index; view < m_cams.size(); view++) {
							availNodes[index][view].clear();
							//printAvailNodes(index, availNodes);
							const auto& epiEdges = m_boneEpiEdges[pafIdx][index - 1][view];
							const int boneAIdx = *std::next(availNodes[index - 1][index - 1].begin(), pick[index - 1]);
							std::cout << "boneAIdx: " << boneAIdx << std::endl;
							const auto& asgnMap = m_assignMap[view];
							for (const int& boneBIdx : availNodes[index - 1][view]) {
								std::cout << "boneBIdx: " << boneBIdx << std::endl;
								if (epiEdges(boneAIdx, boneBIdx) > FLT_EPSILON)
								{
									availNodes[index][view].emplace_back(boneBIdx);
									//printAvailNodes(index, availNodes);
								}


							}
						}
					}
					else
					{
						for (int view = index; view < m_cams.size(); view++)
							availNodes[index][view] = availNodes[index - 1][view];
						std::cout << "set value index: " << index << std::endl;
						printAvailNodes(index, availNodes);
					}
					// temporal constrain
					if (pick[m_cams.size() - 1] >= 0) {
						std::cout << " temporal constrain: " << index << std::endl;
						availNodes[index].back().clear();
						//printAvailNodes(index, availNodes);
						const auto& tempEdge = m_boneTempEdges[pafIdx][m_cams.size() - 1];
						const int boneIdx = *std::next(availNodes[m_cams.size() - 1][m_cams.size() - 1].begin(), pick[m_cams.size() - 1]);
						std::cout << " boneIdx: " << boneIdx << std::endl;
						for (const int& pIdx : availNodes[index - 1].back())
							if (tempEdge(pIdx, boneIdx) > FLT_EPSILON)
							{
								availNodes[index].back().emplace_back(pIdx);
								//printAvailNodes(index, availNodes);
							}
					}
					else
					{
						availNodes[index].back() = availNodes[index - 1].back();
						//printAvailNodes(index, availNodes);
					}

				}
			}
		}
	}
	// combine
	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++)
		cliques.insert(cliques.end(), tmpCliques[pafIdx].begin(), tmpCliques[pafIdx].end());

	std::make_heap(cliques.begin(), cliques.end());
}


void KruskalAssociater::CalcCliqueScore(BoneClique& clique)
{
	// epipolar score
	std::vector<float> scores;
	for (int viewA = 0; viewA < m_cams.size() - 1; viewA++) {
		if (clique.proposal[viewA] == -1)
			continue;
		for (int viewB = viewA + 1; viewB < m_cams.size(); viewB++) {
			if (clique.proposal[viewB] == -1)
				continue;
			scores.emplace_back(m_boneEpiEdges[clique.pafIdx][viewA][viewB](clique.proposal[viewA], clique.proposal[viewB]));
		}
	}

	const float epiScore = scores.empty() ? 1.f
		: std::accumulate(scores.begin(), scores.end(), 0.f) / float(scores.size());

	// temporal score
	scores.clear();
	const int personIdx = clique.proposal[m_cams.size()];
	for (int view = 0; view < m_cams.size() && personIdx != -1; view++) {
		if (clique.proposal[view] == -1)
			continue;
		scores.emplace_back(m_boneTempEdges[clique.pafIdx][view](personIdx, clique.proposal[view]));
	}
	const float tempScore = scores.empty() ? 0.f : std::accumulate(scores.begin(), scores.end(), 0.f) / float(scores.size());

	// paf score
	scores.clear();
	for (int view = 0; view < m_cams.size(); view++) {
		if (clique.proposal[view] == -1)
			continue;
		const Eigen::Vector2i& candi = m_boneNodes[clique.pafIdx][view][clique.proposal[view]];
		scores.emplace_back(m_detections[view].pafs[clique.pafIdx](candi.x(), candi.y()));
	}
	const float pafScore = std::accumulate(scores.begin(), scores.end(), 0.f) / float(scores.size());


	// view score
	const float viewScore = MathUtil::Welsch(m_cViewCnt, float((clique.proposal.head(m_cams.size()).array() >= 0).count()));

	// hier score
	const float hierScore = 1.f - powf(float(m_pafHier[clique.pafIdx]) / float(m_pafHierSize), 4);

	clique.score = (m_wEpi * epiScore + m_wTemp * tempScore + m_wPaf * pafScore + m_wView * viewScore+ m_wHier * hierScore)
		/ (m_wEpi + m_wTemp + m_wPaf + m_wView + m_wHier);
}


void KruskalAssociater::PushClique(const int& pafIdx, const Eigen::VectorXi& proposal, std::vector<BoneClique>& cliques)
{
	if (proposal.head(m_cams.size()).maxCoeff() == -1)
		return;
	BoneClique clique;
	clique.pafIdx = pafIdx;
	clique.proposal = proposal;
	CalcCliqueScore(clique);
	cliques.emplace_back(clique);
	std::push_heap(cliques.begin(), cliques.end());
}


int KruskalAssociater::CheckJointCompatibility(const int& view, const int& jIdx, const int& candiIdx, const int& pIdx)
{
	const SkelDef& def = GetSkelDef(m_type);
	const Eigen::MatrixXi& person = m_personsMap.find(pIdx)->second;
	int checkCnt = 0;
	
	// joint conflict 
	if (person(jIdx, view) != -1 && person(jIdx, view) != candiIdx)
		return -1;

	// paf conflict
	for (const int& pafIdx : m_joint2paf[jIdx]) {
		int checkJIdx = def.pafDict.col(pafIdx).sum() - jIdx;
		if (person(checkJIdx, view) == -1)
			continue;

		int jaCandiIdx = candiIdx;
		int jbCandiIdx = person(checkJIdx, view);
		if (jIdx == def.pafDict(1, pafIdx))
			std::swap(jaCandiIdx, jbCandiIdx);

		if (m_detections[view].pafs[pafIdx](jaCandiIdx, jbCandiIdx) > 0.f)
			checkCnt++;
		else
			return -1;
	}

	// epi conflict
	for (int i = 0; i < m_cams.size(); i++) {
		if (i == view || person(jIdx, i) == -1)
			continue;
		if (m_epiEdges[jIdx][view][i](candiIdx, person(jIdx, i)) > 0.f)
			checkCnt++;
		else
			return -1;
	}

	//// temp conflict
	//if (pIdx < m_skels3dPrev.size()) {
	//	if (m_tempEdges[jIdx][view](pIdx, candiIdx) > 0.f)
	//		checkCnt++;
	//	else
	//		return -1;
	//}

	return checkCnt;
}


int KruskalAssociater::CheckPersonCompatibility(const int& masterIdx, const int& slaveIdx, const int& view)
{
	assert(masterIdx < slaveIdx);
	// master and slave are tracked different temporal person must conflict
	if (slaveIdx < m_skels3dPrev.size())
		return -1;

	int checkCnt = 0;
	const SkelDef& def = GetSkelDef(m_type);
	const Eigen::MatrixXi& master = m_personsMap.find(masterIdx)->second;
	const Eigen::MatrixXi& slave = m_personsMap.find(slaveIdx)->second;

	for (int jIdx = 0; jIdx < def.jointSize; jIdx++)
		if (master(jIdx, view) != -1 && slave(jIdx, view) != -1 && master(jIdx, view) != slave(jIdx, view))
			return -1;

	// master is tracked with temporal person so check the temporal edge
	if (masterIdx < m_skels3dPrev.size())
		for (int jIdx = 0; jIdx < def.jointSize; jIdx++)
			if (slave(jIdx,view) != -1)
				if (m_tempEdges[jIdx][view](masterIdx, slave(jIdx, view)) > 0.f)
					checkCnt++;
				else
					return -1;

	for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++) {
		const Eigen::MatrixXf& paf = m_detections[view].pafs[pafIdx];
		for (const auto& candi : std::vector<Eigen::Vector2i>({
			{master(def.pafDict(0, pafIdx),view), slave(def.pafDict(1, pafIdx),view)},
			{slave(def.pafDict(0, pafIdx),view), master(def.pafDict(1, pafIdx),view)} }))
			if (candi.minCoeff() >= 0)
				if (paf(candi.x(), candi.y()) > 0.f)
					checkCnt++;
				else
					return -1;			// have no paf edge
	}
	return checkCnt;
}

// ������������
int KruskalAssociater::CheckPersonCompatibility(const int& masterIdx, const int& slaveIdx)
{
	assert(masterIdx < slaveIdx);
	if (slaveIdx < m_skels3dPrev.size())
		return -1;

	int checkCnt = 0;
	const SkelDef& def = GetSkelDef(m_type);
	const auto& master = m_personsMap.find(masterIdx)->second;
	const auto& slave = m_personsMap.find(slaveIdx)->second;

	for (int view = 0; view < m_cams.size(); view++) {
		int _checkCnt = CheckPersonCompatibility(masterIdx, slaveIdx, view);
		if (_checkCnt == -1)
			return -1;
		else
			checkCnt += _checkCnt;
	}

	for (int jIdx = 0; jIdx < def.jointSize; jIdx++) {
		for (int viewA = 0; viewA < m_cams.size() - 1; viewA++) {
			const int candiAIdx = master(jIdx, viewA);
			if (candiAIdx != -1) {
				for (int viewB = viewA + 1; viewB < m_cams.size(); viewB++) {
					const int candiBIdx = slave(jIdx, viewB);
					if (candiBIdx != -1)
						if (m_epiEdges[jIdx][viewA][viewB](candiAIdx, candiBIdx) > 0.f)
							checkCnt++;
						else
							return -1;
				}
			}
		}
	}
	return checkCnt;
}


void KruskalAssociater::MergePerson(const int& masterIdx, const int& slaveIdx)
{
	assert(masterIdx < slaveIdx);
	Eigen::MatrixXi& master = m_personsMap.find(masterIdx)->second;
	auto slaveIter = m_personsMap.find(slaveIdx);
	const Eigen::MatrixXi& slave = slaveIter->second;
	const SkelDef& def = GetSkelDef(m_type);
	for (int view = 0; view < m_cams.size(); view++)
		for (int jIdx = 0; jIdx < def.jointSize; jIdx++)
			if (slave(jIdx,view) != -1) {
				master(jIdx,view) = slave(jIdx, view);
				m_assignMap[view][jIdx][slave(jIdx, view)] = masterIdx;
			}
	m_personsMap.erase(slaveIter);
}


void KruskalAssociater::Clique2Voting(const BoneClique& clique, Voting& voting)
{
	voting.vote.clear();
	if (m_personsMap.empty())
		return;
	const SkelDef& def = GetSkelDef(m_type);
	for (int view = 0; view < m_cams.size(); view++) {
		const int index = clique.proposal[view];
		if (index != -1) {
			const Eigen::Vector2i& node = m_boneNodes[clique.pafIdx][view][index];
			for (int i = 0; i < 2; i++) {
				const int assigned = m_assignMap[view][def.pafDict(i, clique.pafIdx)][node[i]];
				if (assigned != -1) {
					auto iter = voting.vote.find(assigned);
					if (iter == voting.vote.end())
						iter = voting.vote.insert(std::make_pair(assigned, Eigen::Vector2i::Zero())).first;
					iter->second[i]++;
				}
			}
		}
	}
}


void KruskalAssociater::Voting::Parse()
{
	fstCnt.setZero();
	secCnt.setZero();
	if (vote.empty())
		return;

	std::map<int, Eigen::Vector2i> _vote = vote;
	for (int i = 0; i < 2; i++) {
		for (int index = 0; index < 2; index++) {
			auto iter = std::max_element(_vote.begin(), _vote.end(), [&index](
				const std::pair<int, Eigen::Vector2i>& l, const std::pair<int, Eigen::Vector2i>& r) {
				return (l.second[index] < r.second[index]);
			});
			
			(i == 0 ? fst : sec)[index] = iter->first;
			(i == 0 ? fstCnt : secCnt)[index] = iter->second[index];
			iter->second[index] = 0;
		}
	}
}


void KruskalAssociater::AssignTopClique(std::vector<BoneClique>& cliques)
{
	const SkelDef& def = GetSkelDef(m_type);
	const BoneClique clique = *cliques.begin();
	std::pop_heap(cliques.begin(), cliques.end());
	cliques.pop_back();

	const auto& nodes = m_boneNodes[clique.pafIdx];
	const auto& jIdxPair = def.pafDict.col(clique.pafIdx);
	if (clique.proposal[m_cams.size()] != -1) {
		const int personIdx = clique.proposal[m_cams.size()];
		const int checkCnt = [&]() {
			int cnt = 0;
			for (int view = 0; view < m_cams.size(); view++) {
				const int index = clique.proposal[view];
				if (index != -1) {
					for (int i = 0; i < 2; i++) {
						const int _cnt = CheckJointCompatibility(view, jIdxPair[i], nodes[view][index][i], personIdx);
						if (_cnt == -1)
							return -1;
						else
							cnt += _cnt;
					}
				}
			}
			return cnt;
		}();

		if (checkCnt != -1) {
			Eigen::MatrixXi& person = m_personsMap.find(personIdx)->second;
			Eigen::VectorXi _proposal = Eigen::VectorXi::Constant(m_cams.size() + 1, -1);
			for (int view = 0; view < m_cams.size(); view++) {
				if (clique.proposal[view] != -1) {
					const Eigen::Vector2i& node = nodes[view][clique.proposal[view]];
					const Eigen::Vector2i assign(m_assignMap[view][jIdxPair.x()][node.x()],
						m_assignMap[view][jIdxPair.y()][node.y()]);
					if ((assign.x() == -1 || assign.x() == personIdx) && (assign.y() == -1 || assign.y() == personIdx)) {
						for (int i = 0; i < 2; i++) {
							person(jIdxPair[i], view) = node[i];
							m_assignMap[view][jIdxPair[i]][node[i]] = personIdx;
						}
					}
					else
						_proposal[view] = clique.proposal[view];
				}
			}
			PushClique(clique.pafIdx, _proposal, cliques);
		}
		else {
			Eigen::VectorXi _proposal = clique.proposal;
			_proposal[m_cams.size()] = -1;
			PushClique(clique.pafIdx, _proposal, cliques);
		}
	}
	else {
		Voting voting;
		Clique2Voting(clique, voting);
		voting.Parse();

		// 1. A & B not assigned yet: 
		if (voting.fstCnt.sum() == 0) {
			const bool allocFlag = [&]() {
				// if the clique is single view, try to associate to existed people
				if ((clique.proposal.array() >= 0).count() == 0)
					return true;
				int view;
				clique.proposal.maxCoeff(&view);
				const Eigen::Vector2i& node = nodes[view][clique.proposal[view]];
				std::vector<std::pair<int, int>> personCandidates;
				for (const auto& person : m_personsMap) {
					const int checkCnt = [&]() {
						int cnt = 0;
						for (int i = 0; i < 2; i++) {
							const int _cnt = CheckJointCompatibility(view, jIdxPair[i], node[i], person.first);
							if (_cnt == -1)
								return -1;
							cnt += _cnt;
						}
						return cnt;
					}();
					if (checkCnt >= m_minCheckCnt)
						personCandidates.emplace_back(std::make_pair(checkCnt, person.first));
				}

				if (personCandidates.size() == 0)
					return true;
				const int personIdx = std::max_element(personCandidates.begin(), personCandidates.end())->second;
				Eigen::MatrixXi& person = m_personsMap.find(personIdx)->second;
				for (int i = 0; i < 2; i++) {
					person(jIdxPair[i], view) = node[i];
					m_assignMap[view][jIdxPair[i]][node[i]] = personIdx;
				}
				return false;
			}();

			// create new person
			if (allocFlag) {
				Eigen::MatrixXi person = Eigen::MatrixXi::Constant(def.jointSize, m_cams.size(), -1);
				const int personIdx = m_personsMap.empty() ? 0 : m_personsMap.rbegin()->first + 1;
				for (int view = 0; view < m_cams.size(); view++) {
					if (clique.proposal[view] >= 0) {
						const Eigen::Vector2i& node = nodes[view][clique.proposal[view]];
						for (int i = 0; i < 2; i++) {
							person(jIdxPair[i], view) = node[i];
							m_assignMap[view][jIdxPair[i]][node[i]] = personIdx;
						}
					}
				}
				m_personsMap.insert(std::make_pair(personIdx, person));
			}
		}

		// 2. A assigned but not B: Add B to person with A (if no another B there) 
		// 3. B assigned but not A: Add A to person with B (if no another A there)
		else if (voting.fstCnt.minCoeff() == 0) {
			const int validIdx = voting.fstCnt[0] > 0 ? 0 : 1;
			const int masterIdx = voting.fst[validIdx];
			const int unasgnJIdx = jIdxPair[1 - validIdx];
			Eigen::MatrixXi& person = m_personsMap.find(masterIdx)->second;

			Eigen::VectorXi _proposal = Eigen::VectorXi::Constant(m_cams.size() + 1, -1);
			for (int view = 0; view < m_cams.size(); view++) {
				if (clique.proposal[view] >= 0) {
					const Eigen::Vector2i& node = nodes[view][clique.proposal[view]];
					const int unasgnJCandiIdx = node[1 - validIdx];
					const int assigned = m_assignMap[view][jIdxPair[validIdx]][node[validIdx]];
					int& unassigned = m_assignMap[view][jIdxPair[1 - validIdx]][node[1 - validIdx]];

					if (assigned == masterIdx) {
						if (person(unasgnJIdx, view) == -1 && CheckJointCompatibility(view, unasgnJIdx, unasgnJCandiIdx, masterIdx) >= 0) {
							person(unasgnJIdx, view) = unasgnJCandiIdx;
							unassigned = masterIdx;
						}
						else
							continue;				// discard this view's propose
					}
					else if (assigned == -1 && voting.fstCnt[validIdx] >= 2 && voting.secCnt[validIdx] == 0
						&& (person(jIdxPair.x(), view) == -1 || person(jIdxPair.x(), view) == node.x())
						&& (person(jIdxPair.y(), view) == -1 || person(jIdxPair.y(), view) == node.y()))
						if (CheckJointCompatibility(view, jIdxPair.x(), node.x(), masterIdx) >= 0
							&& CheckJointCompatibility(view, jIdxPair.y(), node.y(), masterIdx) >= 0) {
							for (int i = 0; i < 2; i++) {
								person(jIdxPair[i], view) = node[i];
								m_assignMap[view][jIdxPair[i]][node[i]] = masterIdx;
							}
						}
						else
							_proposal[view] = clique.proposal[view];
					else
						_proposal[view] = clique.proposal[view];
				}
			}
			if (_proposal != clique.proposal)
				PushClique(clique.pafIdx, _proposal, cliques);
		}

		// 4. A & B already assigned to same person (circular/redundant PAF)
		else if (voting.fst.x() == voting.fst.y()) {
			const int masterIdx = voting.fst.x();
			Eigen::MatrixXi& person = m_personsMap.find(masterIdx)->second;
			Eigen::VectorXi _proposal = Eigen::VectorXi::Constant(m_cams.size() + 1, -1);
			for (int view = 0; view < m_cams.size(); view++) {
				if (clique.proposal[view] >= 0) {
					const Eigen::Vector2i& node = nodes[view][clique.proposal[view]];
					const Eigen::Vector2i assignIdx(m_assignMap[view][jIdxPair.x()][node.x()], m_assignMap[view][jIdxPair.y()][node.y()]);
					if (assignIdx.x() == masterIdx && assignIdx.y() == masterIdx) 
						continue;
					else if (CheckJointCompatibility(view, jIdxPair.x(), node.x(), masterIdx) == -1
						|| CheckJointCompatibility(view, jIdxPair.y(), node.y(), masterIdx) == -1)
						_proposal[view] = clique.proposal[view];

					else if ((assignIdx.x() == masterIdx && assignIdx.y() == -1)
						|| (assignIdx.x() == -1 && assignIdx.y() == masterIdx)) {
						const int validIdx = assignIdx.y() == -1 ? 0 : 1;
						int& unassigned = m_assignMap[view][jIdxPair[1 - validIdx]][node[1 - validIdx]];
						const int unasgnJIdx = jIdxPair[1 - validIdx];
						const int unasgnJCandiIdx = node[1 - validIdx];
						if (person(unasgnJIdx, view) == -1 || person(unasgnJIdx, view) == unasgnJCandiIdx) {
							person(unasgnJIdx, view) = unasgnJCandiIdx;
							unassigned = masterIdx;
						}
						else
							_proposal[view] = clique.proposal[view];
					}
					else if (assignIdx.maxCoeff() == -1 && voting.secCnt.sum() == 0
						&& (person(jIdxPair.x(), view) == -1 || person(jIdxPair.x(), view) == node.x())
						&& (person(jIdxPair.y(), view) == -1 || person(jIdxPair.y(), view) == node.y())) {
						for (int i = 0; i < 2; i++) {
							person(jIdxPair[i], view) = node[i];
							m_assignMap[view][jIdxPair[i]][node[i]] = masterIdx;
						}
					}
					else
						_proposal[view] = clique.proposal[view];
				}
				if (_proposal != clique.proposal)
					PushClique(clique.pafIdx, _proposal, cliques);
			}
		}

		// 5. A & B already assigned to different people: Merge people if key point intersection is null
		else {
			// try to cluster by shared joint
			for (int index = 0; index < 2; index++) {
				while (voting.secCnt[index] != 0) {
					// try to merge master and slave
					const int masterIdx = std::min(voting.fst[index], voting.sec[index]);
					const int slaveIdx = std::max(voting.fst[index], voting.sec[index]);
					if (CheckPersonCompatibility(masterIdx, slaveIdx) >= 0) {
						MergePerson(masterIdx, slaveIdx);
						Clique2Voting(clique, voting);
						voting.Parse();
					}
					else {
						voting.vote.find(voting.fst[index])->second[index] = voting.vote.find(voting.sec[index])->second[index] = 0;
						auto iter = std::max_element(voting.vote.begin(), voting.vote.end(), [&index](
							const std::pair<int, Eigen::Vector2i>& l, const std::pair<int, Eigen::Vector2i>& r) {
							return(l.second[index] < r.second[index]);
						});
						voting.sec[index] = iter->first;
						voting.secCnt[index] = iter->second[index];
					}
				}
			}

			// try to cluster by shared bone
			if (voting.fst.x() != voting.fst.y()) {
				Eigen::VectorXi conflict(m_cams.size());
				const int masterIdx = voting.fst.minCoeff();
				const int slaveIdx = voting.fst.maxCoeff();
				for (int view = 0; view < m_cams.size(); view++)
					conflict[view] = CheckPersonCompatibility(masterIdx, slaveIdx, view) == -1 ? 1 : 0;

				if (conflict.sum() == 0) {
					MergePerson(masterIdx, slaveIdx);

					// proposal unused paf
					Eigen::VectorXi _proposal = Eigen::VectorXi::Constant(m_cams.size() + 1, -1);
					const Eigen::MatrixXi& master = m_personsMap.find(masterIdx)->second;
					for (int view = 0; view < m_cams.size(); view++) {
						if (clique.proposal[view] >= 0) {
							const Eigen::Vector2i& node = nodes[view][clique.proposal[view]];
							if (master(jIdxPair.x(), view) != node.x() || master(jIdxPair.y(), view) != node.y())
								_proposal[view] = clique.proposal[view];
						}
					}
					PushClique(clique.pafIdx, _proposal, cliques);
				}
				else {
					Eigen::MatrixX2i _proposalPair = Eigen::MatrixX2i::Constant(m_cams.size() + 1, 2, -1);
					for (int i = 0; i < conflict.size(); i++)
						_proposalPair(i, conflict[i]) = clique.proposal[i];

					if (_proposalPair.col(0).minCoeff() >= 0 && _proposalPair.col(1).minCoeff() >= 0) {
						PushClique(clique.pafIdx, _proposalPair.col(0), cliques);
						PushClique(clique.pafIdx, _proposalPair.col(1), cliques);
					}
					else if ((clique.proposal.array() >= 0).count() > 1) {
						for (int i = 0; i < conflict.size(); i++) {
							Eigen::VectorXi _proposal = Eigen::VectorXi::Constant(m_cams.size() + 1, -1);
							_proposal[i] = clique.proposal[i];
							PushClique(clique.pafIdx, _proposal, cliques);
						}
					}
				}
			}
		}
	}
}


void KruskalAssociater::DismemberPersons(std::vector<BoneClique>& cliques)
{
	const SkelDef& def = GetSkelDef(m_type);
	for (auto personIter = std::next(m_personsMap.begin(), m_skels3dPrev.size()); personIter != m_personsMap.end(); ) {
		const Eigen::MatrixXi& person = personIter->second;
		if ((person.array() >= 0).count() >= m_minAsgnCnt)
			personIter++;
		else {
			for (int view = 0; view < m_cams.size(); view++) {
				for (int pafIdx = 0; pafIdx < def.pafSize; pafIdx++) {
					const auto jIdxPair = def.pafDict.col(pafIdx);
					const Eigen::Vector2i node(person(jIdxPair.x(), view), person(jIdxPair.y(), view));
					if (node.minCoeff() == -1)
						continue;
					const std::vector<Eigen::Vector2i>& nodes = m_boneNodes[pafIdx][view];
					for (int bone = 0; bone < nodes.size(); bone++)
						if (node == nodes[bone]) {
							Eigen::VectorXi proposal = Eigen::VectorXi::Constant(m_cams.size() + 1, -1);
							proposal[view] = bone;
							PushClique(pafIdx, proposal, cliques);
							break;
						}
				}
			}

			// erase
			for (int view = 0; view < m_cams.size(); view++)
				for (int jIdx = 0; jIdx < def.jointSize; jIdx++)
					if (person(jIdx, view) != -1)
						m_assignMap[view][jIdx][person(jIdx, view)] = -1;
			personIter = m_personsMap.erase(personIter);
		}
	}
}


void KruskalAssociater::SpanTree()
{
	clock_t init_t = clock();
	// ��ʼ�� m_assignMap ��m_personsMap
	Initialize();
	std::cout << " init use time: " << clock() - init_t << std::endl;

	std::vector<BoneClique> cliques;
	clock_t enum_t = clock();
	EnumCliques(cliques);
	std::cout << " enum use time: " << clock() - enum_t << std::endl;

	clock_t empty_t = clock();
	while (!cliques.empty())
		AssignTopClique(cliques);
	std::cout << " empty use time: " << clock() - empty_t << std::endl;

}


void KruskalAssociater::Associate()
{
	clock_t t_1 = clock();
	// �������еĶ�ά�������
	CalcJointRays();
	std::cout << " 1 use time: " << clock() - t_1 << std::endl;
	clock_t t_2 = clock();
	//��һ��pafs �ؽ��׺϶�
	CalcPafEdges();
	std::cout << " 2 use time: " << clock() - t_2 << std::endl;
	clock_t t_3 = clock();
	// �������ж�Ӧ�ؽڵ������ڲ�ͬ��ͼ�еľ��룬��һ���󲢱�����m_epiEdges��
	CalcEpiEdges();
	std::cout << " 3 use time: " << clock() - t_3 << std::endl;
	clock_t t_4 = clock();
	//����֮ǰ����ά�Ǽܵ����ڵ����ж�ά�Ǽ����ߵľ��룬��һ���󱣴浽m_tempEdges �У�����tracking 
	CalcTempEdges();
	std::cout << " 4 use time: " << clock() - t_4 << std::endl;
	clock_t t_5 = clock();
	// �ֱ����ÿ����ͼ�У�ͨ���ؽ���С�׺϶�����Ϲؽ�BoneNodes, �����浽m_boneNodes�У�ͬһ����ͼ
	CalcBoneNodes();
	std::cout << " 5 use time: " << clock() - t_5 << std::endl;
	clock_t t_6 = clock();
	// ����ͼ������ÿһ�Թؽ�boneNodes ͨ����ѯm_epiEdges �ҵ�����ƽ�����롣����ͼ
	CalcBoneEpiEdges();
	std::cout << " 6 use time: " << clock() - t_6 << std::endl;
	clock_t t_7 = clock();
	// ����֮ǰ����ά�Ǽ�BoneNodes ����ǰ���ڹǼܽڵ��ƽ������
	CalcBoneTempEdges();
	std::cout << " 7 use time: " << clock() - t_7 << std::endl;
	clock_t t_8 = clock();
	SpanTree();  // ��Ҫ�ص��ע��
	std::cout << " 8 use time: " << clock() - t_8 << std::endl;
	clock_t t_9 = clock();
	CalcSkels2d();
	std::cout << " 9 use time: " << clock() - t_9 << std::endl;
}
