#include "Loader.hpp"
#include <fstream>
#include <thread>
#include <future>
#include <queue>
#include <map>
#include <unordered_map>
#include <memory>
#include <condition_variable>

#include <filesystem>
namespace  fs = std::filesystem;

namespace thread {
    namespace task {
        using TaskId = size_t;

        class TaskBase {
        public:
            TaskBase() = default;

            TaskBase(const TaskBase&) = delete;
            TaskBase operator=(const TaskBase&) = delete;

            TaskBase(TaskBase&&) = default;
            TaskBase& operator=(TaskBase&&) = default;

            virtual ~TaskBase() = default;
            virtual void Run() = 0;
            virtual TaskId Id() const = 0;
        };

        template <class Result>
        class Task : public TaskBase {
        public:
            virtual Result Get() = 0;
        };

        template <class Result, class F, typename... TaskArgs>
        class TaskImpl : public Task<Result> {
        public:
            TaskImpl(TaskId taskId, F&& f, TaskArgs&&... args)
                : m_id{ taskId }
            {
                m_fut = std::async(std::launch::deferred, std::forward<F>(f), std::forward<TaskArgs>(args)...);
            }

        public:
            Result Get() override {
                return m_fut.get();
            }

            void Run() override {
                m_fut.wait();
            }

            TaskId Id() const override {
                return m_id;
            }

        private:
            TaskId m_id;
            std::future<Result> m_fut;
        };
    }

    class Pool {
    public:
        using TaskId = task::TaskId;

        Pool(int maxNumThreads) {
            m_threads.reserve(maxNumThreads);
            for (int i = 0; i < maxNumThreads; ++i) {
                m_threads.push_back(std::thread(&Pool::Run, this));
            }
        }

        ~Pool() {
            m_stop = true;
            m_queueCond.notify_all();

            for (auto& th : m_threads) {
                if (th.joinable())
                    th.join();
            }
        }

    public:
        template <class Result, typename F, typename... FArgs>
        TaskId AddTask(F&& f, FArgs&&... args) {
            auto task = std::make_unique<task::TaskImpl<Result, F, FArgs...>>(m_nextTaskId, std::forward<F>(f), std::forward<FArgs>(args)...);

            {
                std::lock_guard<std::mutex> guard(m_qTasksMtx);
                m_tasks.push(std::move(task));

                m_queueCond.notify_all();
            }

            return m_nextTaskId++;
        }

        template <class Result>
        Result GetResult(TaskId taskId) {
            std::unique_ptr<task::TaskBase> finishedTask;
            {
                std::lock_guard<std::mutex> guard(m_finishedSetMtx);
                if (m_finishedTasks.find(taskId) == m_finishedTasks.end()) {
                    throw std::logic_error("Pool::GetResult: invalid id has passed as parameter or the task has not been completed.");
                }

                finishedTask = std::move(m_finishedTasks[taskId]);
                m_finishedTasks.erase(taskId);
            }

            task::Task<Result>* t = dynamic_cast<task::Task<Result>*>(finishedTask.get());
            if (!t) {
                throw std::logic_error("Pool::GetResult: invalid type of the result is passed as a template argument.");
            }

            return t->Get();
        }

        void WaitAll() {
            m_requestWaitAll = true;

            std::unique_lock<std::mutex> lock(m_qTasksMtx);
            m_condAllWorkDone.wait(lock, [this]() { return m_tasks.empty() && m_finishedThreads == m_threads.size(); });

            m_requestWaitAll = false;
            m_finishedThreads = 0;
        }

        template <class Result>
        std::vector<Result> GetResults(const std::vector<TaskId>& ids) {
            std::vector<IdResultPair> finishedTasks;

            {
                std::lock_guard<std::mutex> guard(m_finishedSetMtx);

                finishedTasks = ExtractFinishedTasksByIds(ids);
                RemoveFinishedTasks(ids);
            }

            std::vector<Result> results;
            results.reserve(ids.size());

            for (auto& task : finishedTasks) {
                task::Task<Result>* t = dynamic_cast<task::Task<Result>*>(task.second.get());
                if (!t) {
                    throw std::logic_error("Pool::GetResults: invalid type of the result is passed as a template argument.");
                }

                results.push_back(std::move(t->Get()));
            }

            return results;
        }

    private:
        void Run() {
            while (true) {
                std::unique_ptr<task::TaskBase> task;

                {
                    std::unique_lock<std::mutex> lock(m_qTasksMtx);
                    m_queueCond.wait(lock, [this]() { return m_stop || !m_tasks.empty(); });
                    if (m_stop && m_tasks.empty())
                        break;

                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }

                task->Run();

                {
                    std::lock_guard<std::mutex> guard(m_finishedSetMtx);
                    m_finishedTasks[task->Id()] = std::move(task);
                }

                if (m_requestWaitAll) {
                    std::lock_guard<std::mutex> lock(m_qTasksMtx);
                    if (m_tasks.empty()) {
                        m_condAllWorkDone.notify_one();
                        ++m_finishedThreads;
                    }
                }
            }
        }

        using IdResultPair = std::pair <TaskId, std::unique_ptr<task::TaskBase>>;

        std::vector<IdResultPair> ExtractFinishedTasksByIds(const std::vector<size_t>& ids) {
            std::vector<IdResultPair> output;
            output.reserve(ids.size());

            for (const auto id : ids) {
                auto taskIt = m_finishedTasks.find(id);
                if (taskIt == m_finishedTasks.end()) {
                    throw std::logic_error("Pool::GetResults: invalid id has passed as parameter or the task has not been completed.");
                }

                output.emplace_back(id, std::move(taskIt->second));
            }

            return output;
        }

        void RemoveFinishedTasks(const std::vector<size_t>& ids) {
            for (const auto id : ids) {
                m_finishedTasks.erase(id);
            }
        }

    private:
        std::vector<std::thread> m_threads;
        std::atomic<bool> m_stop = false;

        std::condition_variable m_queueCond;
        std::queue<std::unique_ptr<task::TaskBase>> m_tasks;
        std::mutex m_qTasksMtx;

        std::unordered_map<TaskId, std::unique_ptr<task::TaskBase>> m_finishedTasks;
        std::mutex m_finishedSetMtx;

        std::atomic<TaskId> m_nextTaskId = 0;

        std::atomic<bool> m_requestWaitAll = false;
        std::condition_variable m_condAllWorkDone;
        std::atomic<size_t> m_finishedThreads = 0;
    };
}

namespace file {
    Eigen::VectorXf ReadImageVector(const std::string& path) {
        std::ifstream file(path, std::ios::binary);

        const int vec_size = 784;
        Eigen::VectorXf output(vec_size);

        float value = 0.0f;
        int index = 0;
        while (file.read(reinterpret_cast<char*>(&value), sizeof(float))) {
            output(index++) = value;
        }

        file.close();
        return output;
    }

    std::vector<int> ReadYVector(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        
        const int reserve_size = 15000;

        std::vector<int> output;
        output.reserve(reserve_size);

        int value = 0;
        while (file.read(reinterpret_cast<char*>(&value), sizeof(int))) {
            output.push_back(value);
        }

        return output;
    }
}

namespace data {
    enum class DataSet {
        Test,
        Training,
        Validation
    };

    namespace internal {
        std::string DataSetToString(DataSet set) {
            switch (set)
            {
            case data::DataSet::Test:
                return "test_data";
            case data::DataSet::Training:
                return "training_data";
            case data::DataSet::Validation:
                return "validation_data";
            }

            throw std::logic_error("DataSetToString: invalid set name has been passed");
        }

        int ReserveSizeX(DataSet set) {
            const int y_reserve = 1;

            const int test_x_reserve = 10000;
            const int training_x_reserve = 50000;
            const int validation_x_reserve = 10000;

            switch (set)
            {
            case data::DataSet::Test:
                return test_x_reserve;
            case data::DataSet::Training:
                return training_x_reserve;
            case data::DataSet::Validation:
                return validation_x_reserve;
            }

            throw std::logic_error("ReserveSizeX: invalid set name has been passed");
        }

        std::vector<DataSet> AllDataSets() {
            return {
                DataSet::Test,
                DataSet::Training,
                DataSet::Validation
            };
        }
    }

    namespace folder {
        fs::path FindPathOfFolder(const std::string& folder_name) {
            auto path = fs::current_path();
            auto root_path = path.root_path();

            while (path != root_path) {
                for (const auto& entry : fs::directory_iterator(path)) {
                    if (entry.is_directory() && entry.path().filename() == folder_name) {
                        return path / folder_name;
                    }
                }

                path = path.parent_path();
            }
            
            throw std::logic_error("FindPathOfFolder: invalid folder_name has been passed");
        }

        class Manager {
        public:
            Manager(const std::string& folder_name, const std::string& x_folder_name, const std::string& y_file_name)
                : m_root{ folder::FindPathOfFolder(folder_name) }
                , m_x_folder{ x_folder_name }
                , m_y_file{ y_file_name + m_file_format }
            {
            }

        public:
            std::vector<std::string> CreatePathToData(DataSet set) {
                const auto data_path = m_root / internal::DataSetToString(set) / m_x_folder;

                if (!fs::exists(data_path) || !fs::is_directory(data_path)) {
                    throw std::logic_error("Manager::CreatePathToData: the folder either does not exist or not a folder itself");
                }

                std::vector<std::string> file_names;
                file_names.reserve(internal::ReserveSizeX(set));

                for (const auto& file : fs::directory_iterator(data_path)) {
                    if (file.is_regular_file()) {
                        file_names.push_back(file.path().filename().replace_extension().string());
                    }
                }

                // Need a sorted list of file names to match data in "y.txt"
                std::sort(file_names.begin(), file_names.end(), [](const auto& name1, const auto& name2) {
                    return std::atoi(name1.c_str()) < std::atoi(name2.c_str());
                });

                std::vector<std::string> output;
                output.reserve(file_names.size());

                std::transform(file_names.begin(), file_names.end(), std::back_inserter(output), [&data_path, this](const auto& name) {
                    const auto full_name = name + m_file_format;
                    return (data_path / full_name).string();
                });

                return output;
            }

            std::string CreatePathToFile(DataSet set) {
                const auto data_path = m_root / internal::DataSetToString(set);

                if (!fs::exists(data_path) || !fs::is_directory(data_path)) {
                    throw std::logic_error("Manager::CreatePathToFile: the folder either does not exist or not a folder itself");
                }

                for (const auto& file : fs::directory_iterator(data_path)) {
                    if (file.is_regular_file() && file.path().filename() == m_y_file) {
                        return file.path().string();
                    }
                }

                throw std::logic_error("Manager::CreatePathToFile: the file with y-data has not been found");
            }

        private:
            const fs::path m_root;
            const std::string m_file_format = ".bin";
            const std::string m_x_folder;
            const std::string m_y_file;
        };
    }

    namespace vectorize {
        std::vector<Eigen::VectorXf> YResult(std::vector<int> y_vec) {
            std::vector<Eigen::VectorXf> output;
            output.reserve(y_vec.size());

            std::transform(y_vec.begin(), y_vec.end(), std::back_inserter(output), [](const int index) {
                const int size = 10; // numbers from 0 ... 9

                Eigen::VectorXf eigen_vec = Eigen::VectorXf::Zero(size);
                eigen_vec(index) = 1.0f;

                return eigen_vec;
             });

            return output;
        }
    }

    class Loader {
    public:
        Loader(folder::Manager& folder_mgr, int num_threads = 10)
            : m_folder_mgr{ folder_mgr }
            , m_pool(num_threads)
        {
        }

    private:
        struct JobIds {
            std::vector<size_t> m_x_jobs_ids;
            size_t m_y_job_id;
        };

    public:
        Set Load() {
            const auto data_sets = internal::AllDataSets();
            
            std::map<DataSet, JobIds> jobs_ids;
            for (const auto set : data_sets) {
                jobs_ids.emplace(set, RunDataSetJobs(set));
            }
            
            m_pool.WaitAll();

            return {
                Concat(ExtractData(jobs_ids[DataSet::Training])),
                Concat(ExtractData(jobs_ids[DataSet::Validation])),
                Concat(ExtractData(jobs_ids[DataSet::Test]))
            };
        }

    private:
        struct Data {
            std::vector<Eigen::VectorXf> m_x;
            std::vector<Eigen::VectorXf> m_y;
        };

        template <typename Result, typename Func>
        size_t RunJobOnPool(const std::string job, Func&& f) {
            return m_pool.AddTask<Result>(std::forward<Func>(f), job);
        }

        template <typename Result, typename Func>
        std::vector<size_t> RunJobsOnPool(const std::vector<std::string>& jobs, Func&& f) {
            std::vector<size_t> jobs_ids;
            jobs_ids.reserve(jobs.size());

            for (const auto& job : jobs) {
                jobs_ids.push_back(RunJobOnPool<Result>(job, std::forward<Func>(f)));
            }

            return jobs_ids;
        }

        JobIds RunDataSetJobs(DataSet set) {
            const auto x_jobs = m_folder_mgr.CreatePathToData(set);
            const auto y_job = m_folder_mgr.CreatePathToFile(set);

            return {
                RunJobsOnPool<Eigen::VectorXf>(x_jobs, file::ReadImageVector),
                RunJobOnPool<std::vector<int>>(y_job, file::ReadYVector)
            };
        }

        Data ExtractData(const JobIds& ids) {
            return {
                m_pool.GetResults<Eigen::VectorXf>(ids.m_x_jobs_ids),
                vectorize::YResult(m_pool.GetResult<std::vector<int>>(ids.m_y_job_id))
            };
        }

        std::vector<data::PairXY> Concat(Data&& data) {
            assert(data.m_x.size() == data.m_y.size());

            std::vector<data::PairXY> output;
            output.reserve(data.m_x.size());

            std::transform(
                std::make_move_iterator(data.m_x.begin()),
                std::make_move_iterator(data.m_x.end()),
                std::make_move_iterator(data.m_y.begin()),
                std::back_inserter(output),
                [](auto&& x, auto&& y) {
                return data::PairXY{ std::move(x), std::move(y) };
            });

            return output;
        }

    private:
        folder::Manager& m_folder_mgr;
        thread::Pool m_pool;
    };

    namespace loader {
        Set Load() {
            folder::Manager manager("data", "x", "y");
            Loader loader(manager);

            return loader.Load();
        }
    }
}
