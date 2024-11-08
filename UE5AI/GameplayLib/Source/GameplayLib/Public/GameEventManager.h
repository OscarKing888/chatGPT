// EventManager.h
#pragma once

#include "CoreMinimal.h"
#include "Templates/Function.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "GameEventManager.generated.h"

USTRUCT(BlueprintType)
struct FMessageData
{
    GENERATED_BODY()

    // 可以添加必要的事件数据成员
};


// 声明一个通用的事件类型
DECLARE_DYNAMIC_DELEGATE_OneParam(FOnGameEvent, const FMessageData&, Params);

UCLASS()
class GAMEPLAYLIB_API UGameEventManager : public UBlueprintFunctionLibrary
{
    GENERATED_BODY()

public:
    // 使用静态函数注册事件
    UFUNCTION(BlueprintCallable, Category = "Event Manager", meta = (DisplayName = "Register Event"))
        static void RegisterEvent(const FName& EventID, const FOnGameEvent& Callback);

    // 使用静态函数注销事件
    UFUNCTION(BlueprintCallable, Category = "Event Manager")
        static void UnregisterEvent(const FName& EventID);

    // 使用静态函数触发事件
    UFUNCTION(BlueprintCallable, Category = "Event Manager")
        static void TriggerEvent(const FName& EventID, const FMessageData& Params);

private:
    // 存储事件订阅者的TMap
    static TMap<FName, FOnGameEvent> EventMap;
};
