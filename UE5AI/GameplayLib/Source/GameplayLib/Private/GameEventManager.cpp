// EventManager.cpp
#include "GameEventManager.h"

TMap<FName, FOnGameEvent> UGameEventManager::EventMap;

void UGameEventManager::RegisterEvent(const FName& EventID, const FOnGameEvent& Callback)
{
    // 将事件添加到事件映射
    EventMap.Add(EventID, Callback);
}

void UGameEventManager::UnregisterEvent(const FName& EventID)
{
    // 移除事件
    EventMap.Remove(EventID);
}

void UGameEventManager::TriggerEvent(const FName& EventID, const FMessageData& Params)
{
    // 查找事件
    FOnGameEvent* EventDelegate = EventMap.Find(EventID);
    if (EventDelegate)
    {
        // 触发事件
        EventDelegate->ExecuteIfBound(Params);
    }
}
